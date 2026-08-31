from __future__ import annotations

import copy
import multiprocessing as mp
import pickle
import random
import time
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, Generator, List, Optional, Tuple

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationDAGBuilder,
)
from systemds.scuro.drsearch.task import Task
from systemds.scuro.modality.modality import Modality
from systemds.scuro.utils.checkpointing import CheckpointManager
from systemds.scuro.utils.static_variables import DEBUG

# ----------------------------
# Genome / Individual encoding
# ----------------------------

Tree = Any  # int leaf index OR tuple(left_subtree, right_subtree)


def genome_to_dag(genome) -> RepresentationDag:
    builder = RepresentationDAGBuilder()
    leaf_ids = [
        builder.create_leaf_node(mod_id, repr_idx) for mod_id, repr_idx in genome.leaves
    ]

    def build(subtree: Tree, path: str) -> str:
        if isinstance(subtree, int):
            return leaf_ids[subtree]
        left, right = subtree
        left_id = build(left, path + "L")
        right_id = build(right, path + "R")
        op_cls = genome.fusion_ops[path]
        op = op_cls()
        return builder.create_operation_node(
            op.__class__, [left_id, right_id], op.get_current_parameters()
        )

    return builder.build(build(genome.tree, ""))


def _evaluate_individual_worker(
    dag_pickle: bytes,
    task_pickle: bytes,
    modalities_pickle: bytes,
    metric_name: str,
) -> Tuple[float, Dict[str, Any]]:
    dag = pickle.loads(dag_pickle)
    task = pickle.loads(task_pickle)
    modalities = pickle.loads(modalities_pickle)

    start_time = time.time()
    fused_representation = dag.execute(modalities, task)
    scores = task.run(fused_representation.data)
    runtime = time.time() - start_time
    fitness = scores[1].average_scores[metric_name]

    objective = {
        "train_score": scores[0].average_scores,
        "val_score": scores[1].average_scores,
        "test_score": scores[2].average_scores,
        "val": fitness,
        "runtime": runtime,
        "representation_time": runtime,
        "task_time": 0.0,
    }
    return fitness, objective


@dataclass
class Individual:
    # Leaves are concrete unimodal choices: (modality_id, representation_index)
    leaves: List[Tuple[str, int]]
    # Binary tree over leaf indices in `leaves`
    tree: Tree
    # Fusion op class per internal tree-path key, e.g. "L", "RLL", ...
    fusion_ops: Dict[str, Any]
    fitness: float = float("-inf")
    objective: Dict[str, float] = field(default_factory=dict)
    dag: Optional[RepresentationDag] = None


# ----------------------------
# Optional helper abstraction
# ----------------------------


class MutationOperator:
    name: str = "base"

    def __call__(
        self,
        individual: Individual,
        rng: random.Random,
        context: Dict[str, Any],
    ) -> Individual:
        raise NotImplementedError()


def _collect_internal_paths(subtree: Tree, path: str = "") -> List[str]:
    if isinstance(subtree, int):
        return []
    left, right = subtree
    return (
        [path]
        + _collect_internal_paths(left, path + "L")
        + _collect_internal_paths(right, path + "R")
    )


def _get_subtree(subtree: Tree, target_path: str) -> Tree:
    if target_path == "":
        return copy.deepcopy(subtree)
    if isinstance(subtree, int):
        raise ValueError(f"Path '{target_path}' does not exist in leaf subtree")
    left, right = subtree
    if target_path[0] == "L":
        return _get_subtree(left, target_path[1:])
    return _get_subtree(right, target_path[1:])


def _replace_subtree(subtree: Tree, target_path: str, replacement: Tree) -> Tree:
    if target_path == "":
        return copy.deepcopy(replacement)
    if isinstance(subtree, int):
        raise ValueError(f"Path '{target_path}' does not exist in leaf subtree")
    left, right = subtree
    if target_path[0] == "L":
        return (
            _replace_subtree(left, target_path[1:], replacement),
            copy.deepcopy(right),
        )
    return (copy.deepcopy(left), _replace_subtree(right, target_path[1:], replacement))


def _collect_leaf_indices(subtree: Tree) -> List[int]:
    if isinstance(subtree, int):
        return [subtree]
    left, right = subtree
    return _collect_leaf_indices(left) + _collect_leaf_indices(right)


def _sample_random_binary_tree_from_leaves(
    leaf_indices: List[int], rng: random.Random
) -> Tree:
    nodes: List[Tree] = list(leaf_indices)
    while len(nodes) > 1:
        i, j = rng.sample(range(len(nodes)), 2)
        a = nodes.pop(max(i, j))
        b = nodes.pop(min(i, j))
        nodes.append((a, b))
    return nodes[0]


def _reindex_tree(subtree: Tree, index_map: Dict[int, int]) -> Tree:
    if isinstance(subtree, int):
        return index_map[subtree]
    left, right = subtree
    return (_reindex_tree(left, index_map), _reindex_tree(right, index_map))


def _remove_leaf_from_tree(subtree: Tree, leaf_idx: int) -> Optional[Tree]:
    if isinstance(subtree, int):
        if subtree == leaf_idx:
            return None
        return subtree

    left, right = subtree
    new_left = _remove_leaf_from_tree(left, leaf_idx)
    new_right = _remove_leaf_from_tree(right, leaf_idx)

    if new_left is None:
        return new_right
    if new_right is None:
        return new_left
    return (new_left, new_right)


def _path_is_under_prefix(path: str, prefix: str) -> bool:
    return prefix == "" or path == prefix or path.startswith(prefix)


def _rebuild_fusion_ops(
    subtree: Tree,
    existing_fusion_ops: Dict[str, Any],
    rng: random.Random,
    fusion_operators: List[Any],
    randomized_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    fusion_ops: Dict[str, Any] = {}
    randomized_prefixes = randomized_prefixes or []
    for path in _collect_internal_paths(subtree):
        if any(_path_is_under_prefix(path, prefix) for prefix in randomized_prefixes):
            fusion_ops[path] = rng.choice(fusion_operators)
        elif path in existing_fusion_ops:
            fusion_ops[path] = existing_fusion_ops[path]
        else:
            fusion_ops[path] = rng.choice(fusion_operators)
    return fusion_ops


# ----------------------------
# GA Optimizer skeleton
# ----------------------------


class MultimodalGAOptimizer:
    """
    GA-based multimodal optimizer skeleton.
    Keeps constructor inputs compatible with original MultimodalOptimizer.
    """

    def __init__(
        self,
        modalities: List[Any],
        unimodal_optimization_results: Any,
        tasks: List[Any],
        k: int = 2,
        debug: bool = False,
        min_modalities: int = 2,
        max_modalities: int = None,
        metric: str = "accuracy",
        checkpoint_every: int = None,
        resume: bool = True,
        # --- GA controls (new) ---
        population_size: int = 32,
        generations: int = 20,
        elite_size: int = 4,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.3,
        random_seed: int = 42,
        early_stopping_patience: Optional[int] = 5,
        early_stopping_min_delta: float = 1e-6,
    ):
        self.modalities = modalities
        self.tasks = tasks
        self.k = k
        self.debug = debug
        if DEBUG:
            self.debug = True
        self.metric_name = metric
        self.min_modalities = max(2, min_modalities)
        self.max_modalities = max_modalities or len(modalities)

        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.operator_registry = Registry()
        self.fusion_operators = self.operator_registry.get_fusion_operators()

        self.k_best_representations = self._extract_k_best_representations(
            unimodal_optimization_results
        )

        self.optimization_results: Dict[str, List[OptimizationResult]] = {}
        self._seen_results_by_task: Dict[str, Dict[str, OptimizationResult]] = {}
        self._fitness_cache_by_task: Dict[
            str, Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]]
        ] = {}
        self._stats_by_task: Dict[str, Dict[str, int]] = {}
        self._checkpoint_manager = CheckpointManager(
            ".",
            "multimodal_ga_checkpoint_",
            checkpoint_every=checkpoint_every,
            resume=resume,
        )

        # Register mutation ops you want enabled
        self.mutation_ops: List[MutationOperator] = [
            MutateFusionOperator(),
            MutateRepresentationIndex(),
            MutateTreeRotation(),
            MutateSubtreeResample(),
            # Optional global search mutations:
            MutateAddModality(),
            MutateDropModality(),
        ]

    # ----------------------------
    # Public entrypoint
    # ----------------------------

    def optimize(
        self,
        max_evaluations_per_task: Optional[int] = None,
        # If you later wire this to NodeExecutor, add params like:
        # use_node_executor: bool = True,
        # max_workers: int = ...
    ) -> Dict[str, List[OptimizationResult]]:
        self.rng = random.Random(self.random_seed)
        self._resume_if_available()
        self._ensure_results_initialized()

        for task in self.tasks:
            task_name = task.model.name
            if self.debug:
                print(f"[GA] Task={task_name} initialization")

            # 1) Initialize population stochastically
            population = self._initialize_population(task_name)

            eval_budget = 0
            best_seen: Optional[Individual] = None
            no_improvement_gens = 0

            # 2) Evolution loop
            for gen in range(self.generations):
                if self.debug:
                    print(f"[GA] Task={task_name}, generation={gen}")

                # Evaluate all individuals (or only unevaluated)
                self._evaluate_population(population, task)

                # Track best
                gen_best = max(population, key=lambda ind: ind.fitness)
                if best_seen is None or (
                    gen_best.fitness > best_seen.fitness + self.early_stopping_min_delta
                ):
                    best_seen = copy.deepcopy(gen_best)
                    no_improvement_gens = 0
                else:
                    no_improvement_gens += 1

                # Optional budget stop
                eval_budget += self._stats_by_task[task_name][
                    "last_generation_executed"
                ]
                if (
                    max_evaluations_per_task is not None
                    and eval_budget >= max_evaluations_per_task
                ):
                    break
                if (
                    self.early_stopping_patience is not None
                    and no_improvement_gens >= self.early_stopping_patience
                ):
                    if self.debug:
                        print(
                            f"[GA] Task={task_name} early stopping after "
                            f"{no_improvement_gens} no-improvement generations"
                        )
                    break

                # Selection + variation + replacement
                next_population = self._elitism(population)
                seen_genotypes = {
                    self._individual_signature(ind) for ind in next_population
                }
                duplicate_retry_budget = self.population_size * 10
                duplicate_retries = 0
                while len(next_population) < self.population_size:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)

                    if self.rng.random() < self.crossover_rate:
                        c1, c2 = self._crossover(p1, p2)
                    else:
                        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

                    c1 = self._mutate(c1, task_name)
                    c2 = self._mutate(c2, task_name)

                    added_c1 = self._append_if_unique(
                        next_population, c1, seen_genotypes, self.population_size
                    )
                    added_c2 = self._append_if_unique(
                        next_population, c2, seen_genotypes, self.population_size
                    )
                    if not added_c1 and not added_c2:
                        duplicate_retries += 1
                    else:
                        duplicate_retries = 0

                    if duplicate_retries >= duplicate_retry_budget:
                        break

                while len(next_population) < self.population_size:
                    immigrant = self._sample_random_individual(task_name)
                    self._append_if_unique(
                        next_population, immigrant, seen_genotypes, self.population_size
                    )

                population = next_population

            self._checkpoint_manager.checkpoint_if_due(
                self.optimization_results, "eval_count_by_task"
            )

            if self.debug and best_seen is not None:
                print(f"[GA] Task={task_name}, best_fitness={best_seen.fitness:.6f}")

        return self.optimization_results

    def optimize_parallel(
        self,
        max_combinations: Optional[int] = None,
        max_workers: int = 2,
        batch_size: int = 8,
    ) -> Dict[str, List[OptimizationResult]]:
        self.rng = random.Random(self.random_seed)
        self._resume_if_available()
        self._ensure_results_initialized()

        for task in self.tasks:
            task_name = task.model.name
            if self.debug:
                print(f"[GA-P] Task={task_name} initialization")

            population = self._initialize_population(task_name)
            eval_budget = 0
            best_seen: Optional[Individual] = None
            no_improvement_gens = 0

            for gen in range(self.generations):
                if self.debug:
                    print(f"[GA-P] Task={task_name}, generation={gen}")

                self._evaluate_population_parallel(
                    population,
                    task,
                    max_workers=max_workers,
                    batch_size=max(1, batch_size),
                )

                gen_best = max(population, key=lambda ind: ind.fitness)
                if best_seen is None or (
                    gen_best.fitness > best_seen.fitness + self.early_stopping_min_delta
                ):
                    best_seen = copy.deepcopy(gen_best)
                    no_improvement_gens = 0
                else:
                    no_improvement_gens += 1

                eval_budget += self._stats_by_task[task_name][
                    "last_generation_executed"
                ]
                if max_combinations is not None and eval_budget >= max_combinations:
                    break
                if (
                    self.early_stopping_patience is not None
                    and no_improvement_gens >= self.early_stopping_patience
                ):
                    if self.debug:
                        print(
                            f"[GA-P] Task={task_name} early stopping after "
                            f"{no_improvement_gens} no-improvement generations"
                        )
                    break

                next_population = self._elitism(population)
                seen_genotypes = {
                    self._individual_signature(ind) for ind in next_population
                }
                duplicate_retry_budget = self.population_size * 10
                duplicate_retries = 0
                while len(next_population) < self.population_size:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)

                    if self.rng.random() < self.crossover_rate:
                        c1, c2 = self._crossover(p1, p2)
                    else:
                        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

                    c1 = self._mutate(c1, task_name)
                    c2 = self._mutate(c2, task_name)

                    added_c1 = self._append_if_unique(
                        next_population, c1, seen_genotypes, self.population_size
                    )
                    added_c2 = self._append_if_unique(
                        next_population, c2, seen_genotypes, self.population_size
                    )
                    if not added_c1 and not added_c2:
                        duplicate_retries += 1
                    else:
                        duplicate_retries = 0

                    if duplicate_retries >= duplicate_retry_budget:
                        break

                while len(next_population) < self.population_size:
                    immigrant = self._sample_random_individual(task_name)
                    self._append_if_unique(
                        next_population, immigrant, seen_genotypes, self.population_size
                    )

                population = next_population

            self._checkpoint_manager.checkpoint_if_due(
                self.optimization_results, "eval_count_by_task"
            )

            if self.debug and best_seen is not None:
                print(f"[GA-P] Task={task_name}, best_fitness={best_seen.fitness:.6f}")

        return self.optimization_results

    # ----------------------------
    # Initialization
    # ----------------------------

    def _initialize_population(self, task_name: str) -> List[Individual]:
        population = []
        seen_signatures = set()
        retry_budget = self.population_size * 10
        retries = 0
        for _ in range(self.population_size):
            while retries < retry_budget:
                candidate = self._sample_random_individual(task_name)
                signature = self._individual_signature(candidate)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    population.append(candidate)
                    break
                retries += 1
            if retries >= retry_budget:
                # Fall back to potentially duplicated random individuals rather than
                # stalling initialization on very small search spaces.
                population.append(self._sample_random_individual(task_name))
        return population

    def _sample_random_individual(self, task_name: str) -> Individual:
        leaves = self._sample_leaf_set(task_name)
        tree = self._sample_random_binary_tree(len(leaves))
        fusion_ops = {}
        self._assign_random_fusion_ops(tree, fusion_ops, path="")
        return Individual(leaves=leaves, tree=tree, fusion_ops=fusion_ops)

    def _sample_leaf_set(self, task_name: str) -> List[Tuple[str, int]]:
        # Only sample from modalities that have at least one representation
        # for the current task.
        task_reps = self.k_best_representations.get(task_name, {})
        available_modalities = [
            m.modality_id
            for m in self.modalities
            if len(task_reps.get(m.modality_id, [])) > 0
        ]

        if len(available_modalities) < 2:
            raise ValueError(
                f"Need at least 2 modalities with non-empty representations for task "
                f"'{task_name}', found {len(available_modalities)}."
            )

        # Clamp sampling range to available modalities.
        lower = max(2, self.min_modalities)
        upper = min(self.max_modalities, len(available_modalities))
        if lower > upper:
            lower = upper

        r = self.rng.randint(lower, upper)
        chosen_modalities = self.rng.sample(available_modalities, r)

        # Sample one representation index from top-k for each chosen modality.
        return [
            (mod_id, self.rng.randrange(len(task_reps[mod_id])))
            for mod_id in chosen_modalities
        ]

    def _sample_random_binary_tree(self, n_leaves: int) -> Tree:
        nodes: List[Tree] = list(range(n_leaves))
        while len(nodes) > 1:
            i, j = self.rng.sample(range(len(nodes)), 2)
            a = nodes.pop(max(i, j))
            b = nodes.pop(min(i, j))
            nodes.append((a, b))
        return nodes[0]

    def _assign_random_fusion_ops(
        self, subtree: Tree, fusion_ops: Dict[str, Any], path: str
    ):
        if isinstance(subtree, int):
            return
        fusion_ops[path] = self.rng.choice(self.fusion_operators)
        left, right = subtree
        self._assign_random_fusion_ops(left, fusion_ops, path + "L")
        self._assign_random_fusion_ops(right, fusion_ops, path + "R")

    # ----------------------------
    # Evaluation
    # ----------------------------

    def _evaluate_population(self, population: List[Individual], task: Task) -> None:
        # Placeholder: currently serial + per-individual eval.
        # You can batch DAGs and evaluate with NodeExecutor later.
        task_name = task.model.name
        cache = self._fitness_cache_by_task.setdefault(task_name, {})
        evals_before = self._stats_by_task[task_name]["executed_evaluations"]
        for ind in population:
            # Already evaluated and unchanged.
            if ind.fitness != float("-inf"):
                continue

            signature = self._individual_signature(ind)
            cached = cache.get(signature)
            if cached is not None:
                self._stats_by_task[task_name]["cache_hits"] += 1
                ind.fitness, ind.objective = cached[0], dict(cached[1])
                continue

            dag = self._individual_to_dag(ind)
            ind.dag = dag
            modalities = [
                self.k_best_representations[task_name][mod_id][repr_idx]
                for mod_id, repr_idx in ind.leaves
            ]
            self._stats_by_task[task_name]["executed_evaluations"] += 1
            fitness, objective = self._evaluate_individual(dag, task, modalities)
            ind.fitness = fitness
            ind.objective = objective
            cache[signature] = (fitness, dict(objective))
            self._record_result(task_name, dag, objective)
        self._stats_by_task[task_name]["last_generation_executed"] = (
            self._stats_by_task[task_name]["executed_evaluations"] - evals_before
        )

    def _evaluate_population_parallel(
        self,
        population: List[Individual],
        task: Task,
        max_workers: int,
        batch_size: int,
    ) -> None:
        ctx = mp.get_context("spawn")
        task_name = task.model.name
        cache = self._fitness_cache_by_task.setdefault(task_name, {})
        evals_before = self._stats_by_task[task_name]["executed_evaluations"]
        task_pickle = pickle.dumps(copy.deepcopy(task))
        futures = {}
        pending_followers: Dict[Tuple[Any, ...], List[Individual]] = {}

        def _collect_ready(done_futures):
            for done in done_futures:
                ind, dag, signature = futures.pop(done)
                fitness, objective = done.result()
                ind.fitness = fitness
                ind.objective = objective
                cache[signature] = (fitness, dict(objective))
                self._record_result(task_name, dag, objective)
                for follower in pending_followers.pop(signature, []):
                    follower.fitness = fitness
                    follower.objective = dict(objective)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for ind in population:
                # Already evaluated and unchanged.
                if ind.fitness != float("-inf"):
                    continue

                signature = self._individual_signature(ind)
                cached = cache.get(signature)
                if cached is not None:
                    self._stats_by_task[task_name]["cache_hits"] += 1
                    ind.fitness, ind.objective = cached[0], dict(cached[1])
                    continue

                if signature in pending_followers:
                    pending_followers[signature].append(ind)
                    self._stats_by_task[task_name]["pending_signature_hits"] += 1
                    continue

                dag = self._individual_to_dag(ind)
                ind.dag = dag
                modalities = [
                    self.k_best_representations[task_name][mod_id][repr_idx]
                    for mod_id, repr_idx in ind.leaves
                ]

                fut = executor.submit(
                    _evaluate_individual_worker,
                    pickle.dumps(dag),
                    task_pickle,
                    pickle.dumps(modalities),
                    self.metric_name,
                )
                futures[fut] = (ind, dag, signature)
                pending_followers[signature] = []
                self._stats_by_task[task_name]["executed_evaluations"] += 1

                if len(futures) >= batch_size:
                    done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                    _collect_ready(done)

            if futures:
                done, _ = wait(set(futures.keys()))
                _collect_ready(done)
        self._stats_by_task[task_name]["last_generation_executed"] = (
            self._stats_by_task[task_name]["executed_evaluations"] - evals_before
        )

    def _evaluate_individual(
        self, dag: RepresentationDag, task: Task, modalities: List[Modality]
    ) -> Tuple[float, Dict[str, float]]:
        start_time = time.time()
        fused_representation = dag.execute(modalities, task)
        task_start_time = time.time()
        scores = task.run(fused_representation.data)
        task_end_time = time.time()
        runtime = time.time() - start_time
        fitness = scores[1].average_scores[self.metric_name]
        objective = {
            "train_score": scores[0].average_scores,
            "val_score": scores[1].average_scores,
            "test_score": scores[2].average_scores,
            "val": fitness,
            "runtime": runtime,
            "representation_time": fused_representation.transform_time,
            "task_time": task_end_time - task_start_time,
        }
        return fitness, objective

    def _record_result(
        self, task_name: str, dag: RepresentationDag, objective: Dict[str, Any]
    ) -> None:
        key = str(dag.compute_full_node_signature(dag.root_node_id))
        if key in self._seen_results_by_task[task_name]:
            return
        result = OptimizationResult(
            dag=dag,
            val_score=objective.get("val_score", {}),
            train_score=objective.get("train_score", {}),
            test_score=objective.get("test_score", {}),
            runtime=objective.get("runtime", 0.0),
            representation_time=objective.get("representation_time", 0.0),
            task_time=objective.get("task_time", 0.0),
            task_name=task_name,
        )
        self._seen_results_by_task[task_name][key] = result
        self.optimization_results[task_name].append(result)

    def get_task_stats(self, task_name: str) -> Dict[str, int]:
        return dict(self._stats_by_task.get(task_name, {}))

    def _individual_signature(self, ind: Individual) -> Tuple[Any, ...]:
        def _normalize_tree(subtree: Tree) -> Any:
            if isinstance(subtree, int):
                return subtree
            left, right = subtree
            return (_normalize_tree(left), _normalize_tree(right))

        leaves_sig = tuple(ind.leaves)
        tree_sig = _normalize_tree(ind.tree)
        ops_sig = tuple(
            sorted((path, op.__name__) for path, op in ind.fusion_ops.items())
        )
        return leaves_sig, tree_sig, ops_sig

    def _append_if_unique(
        self,
        population: List[Individual],
        candidate: Individual,
        seen_genotypes: set,
        target_size: int,
    ) -> bool:
        if len(population) >= target_size:
            return False
        sig = self._individual_signature(candidate)
        if sig in seen_genotypes:
            return False
        seen_genotypes.add(sig)
        population.append(candidate)
        return True

    # ----------------------------
    # Selection / crossover / mutation
    # ----------------------------

    def _elitism(self, population: List[Individual]) -> List[Individual]:
        ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
        return [copy.deepcopy(ind) for ind in ranked[: self.elite_size]]

    def _tournament_select(self, population: List[Individual]) -> Individual:
        contestants = self.rng.sample(
            population, k=min(self.tournament_size, len(population))
        )
        return copy.deepcopy(max(contestants, key=lambda x: x.fitness))

    def _crossover(
        self, p1: Individual, p2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Skeleton crossover:
        - Safe version assumes compatible leaves (same modality set/order).
        - If incompatible, fallback to op-only crossover.
        """
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

        if self._compatible_for_subtree_crossover(c1, c2):
            c1_paths = _collect_internal_paths(c1.tree)
            c2_paths = _collect_internal_paths(c2.tree)
            if c1_paths and c2_paths:
                path1 = self.rng.choice(c1_paths)
                path2 = self.rng.choice(c2_paths)
                subtree1 = _get_subtree(c1.tree, path1)
                subtree2 = _get_subtree(c2.tree, path2)
                c1.tree = _replace_subtree(c1.tree, path1, subtree2)
                c2.tree = _replace_subtree(c2.tree, path2, subtree1)
                c1.fusion_ops = _rebuild_fusion_ops(
                    c1.tree,
                    c1.fusion_ops,
                    self.rng,
                    self.fusion_operators,
                    randomized_prefixes=[path1],
                )
                c2.fusion_ops = _rebuild_fusion_ops(
                    c2.tree,
                    c2.fusion_ops,
                    self.rng,
                    self.fusion_operators,
                    randomized_prefixes=[path2],
                )
        else:
            # op-only crossover: mix operator assignments
            all_keys = set(c1.fusion_ops.keys()) | set(c2.fusion_ops.keys())
            for k in all_keys:
                if self.rng.random() < 0.5:
                    if k in c2.fusion_ops:
                        c1.fusion_ops[k] = c2.fusion_ops[k]
                else:
                    if k in c1.fusion_ops:
                        c2.fusion_ops[k] = c1.fusion_ops[k]

        # invalidate stale eval
        c1.fitness, c1.objective, c1.dag = float("-inf"), {}, None
        c2.fitness, c2.objective, c2.dag = float("-inf"), {}, None
        return c1, c2

    def _compatible_for_subtree_crossover(self, a: Individual, b: Individual) -> bool:
        return [m for m, _ in a.leaves] == [m for m, _ in b.leaves] and len(
            a.leaves
        ) == len(b.leaves)

    def _mutate(self, ind: Individual, task_name: str) -> Individual:
        out = copy.deepcopy(ind)
        if self.rng.random() >= self.mutation_rate:
            return out

        op = self.rng.choice(self.mutation_ops)
        context = {
            "task_name": task_name,
            "fusion_operators": self.fusion_operators,
            "k_best_representations": self.k_best_representations,
            "min_modalities": self.min_modalities,
            "max_modalities": self.max_modalities,
            "all_modalities": [m.modality_id for m in self.modalities],
        }
        out = op(out, self.rng, context)
        out.fitness, out.objective, out.dag = float("-inf"), {}, None
        return out

    # ----------------------------
    # Genome -> DAG
    # ----------------------------

    def _individual_to_dag(self, ind: Individual) -> RepresentationDag:
        builder = RepresentationDAGBuilder()
        leaf_ids = []
        for modality_id, repr_idx in ind.leaves:
            # Leaves reference the cached transformed modality directly.
            # We collapse the unimodal path from raw->representation here, so
            # downstream execution only needs to run multimodal fusion nodes.
            leaf_ids.append(builder.create_leaf_node(modality_id, repr_idx))

        def build(subtree: Tree, path: str) -> str:
            if isinstance(subtree, int):
                return leaf_ids[subtree]
            left, right = subtree
            left_id = build(left, path + "L")
            right_id = build(right, path + "R")
            op_cls = ind.fusion_ops[path]
            op = op_cls()
            return builder.create_operation_node(
                op.__class__, [left_id, right_id], op.get_current_parameters()
            )

        root_id = build(ind.tree, "")
        dag = builder.build(root_id)
        return self._collapse_cached_unimodal_nodes(dag)

    def _collapse_cached_unimodal_nodes(
        self, dag: RepresentationDag
    ) -> RepresentationDag:
        """
        Remove unary unimodal nodes that originate at leaves.

        In GA multimodal search, leaf inputs are already transformed unimodal
        representations taken from the unimodal optimizer cache. Any unary chain
        attached to those leaves is redundant and can be bypassed to ensure only
        multimodal (fusion) operations are executed.
        """
        node_by_id = {node.node_id: copy.deepcopy(node) for node in dag.nodes}
        if dag.root_node_id not in node_by_id:
            return dag

        changed = True
        while changed:
            changed = False
            node_ids = list(node_by_id.keys())
            for node_id in node_ids:
                if node_id not in node_by_id:
                    continue
                node = node_by_id[node_id]
                if len(node.inputs) != 1:
                    continue

                parent_id = node.inputs[0]
                parent = node_by_id.get(parent_id)
                if parent is None:
                    continue
                if parent.inputs:
                    continue

                # Bypass unary node by rewiring all consumers to the leaf input.
                for consumer in node_by_id.values():
                    consumer.inputs = [
                        parent_id if input_id == node_id else input_id
                        for input_id in consumer.inputs
                    ]
                if dag.root_node_id == node_id:
                    dag.root_node_id = parent_id
                del node_by_id[node_id]
                changed = True

        return RepresentationDag(
            list(node_by_id.values()), dag.root_node_id, dag.dag_id
        )

    # ----------------------------
    # Existing helper compatibility
    # ----------------------------

    def _extract_k_best_representations(
        self, unimodal_optimization_results: Any
    ) -> Dict[str, Dict[str, List[Any]]]:
        k_best = {}
        for task in self.tasks:
            task_name = task.model.name
            k_best[task_name] = {}
            for modality in self.modalities:
                _, cached_data = unimodal_optimization_results.get_k_best_results(
                    modality, task, self.metric_name
                )
                k_best[task_name][modality.modality_id] = cached_data
        return k_best

    def _resume_if_available(self) -> None:
        loaded = self._checkpoint_manager.resume_from_checkpoint(
            "eval_count_by_task",
            lambda results: {
                t.model.name: len(results.get(t.model.name, [])) for t in self.tasks
            },
        )
        if loaded:
            results, _, _ = loaded
            self.optimization_results = results

    def _ensure_results_initialized(self):
        if not isinstance(self.optimization_results, dict):
            self.optimization_results = {}
        for task in self.tasks:
            task_name = task.model.name
            self.optimization_results.setdefault(task_name, [])
            seen = self._seen_results_by_task.setdefault(task_name, {})
            self._fitness_cache_by_task.setdefault(task_name, {})
            self._stats_by_task.setdefault(
                task_name,
                {
                    "executed_evaluations": 0,
                    "last_generation_executed": 0,
                    "cache_hits": 0,
                    "pending_signature_hits": 0,
                },
            )
            if self.optimization_results[task_name]:
                for result in self.optimization_results[task_name]:
                    if result.dag is None:
                        continue
                    key = str(
                        result.dag.compute_full_node_signature(result.dag.root_node_id)
                    )
                    seen.setdefault(key, result)

    def _to_optimization_results(
        self, individuals: List[Individual], task_name: str
    ) -> List[OptimizationResult]:
        out = []
        for ind in individuals:
            # Keep consistent with old output shape: OptimizationResult list per task
            out.append(
                OptimizationResult(
                    dag=ind.dag,
                    val_score={
                        self.metric_name: ind.objective.get("val", float("-inf"))
                    },
                    train_score={},
                    test_score={},
                    runtime=ind.objective.get("runtime", 0.0),
                    representation_time=ind.objective.get("representation_time", 0.0),
                    task_time=ind.objective.get("task_time", 0.0),
                    task_name=task_name,
                )
            )
        return out


# ----------------------------
# Mutation operators (outline)
# ----------------------------


class MutateFusionOperator(MutationOperator):
    """
    Change one internal fusion op class.
    """

    name = "mutate_fusion_operator"

    def __call__(
        self, individual: Individual, rng: random.Random, context: Dict[str, Any]
    ) -> Individual:
        out = copy.deepcopy(individual)
        if not out.fusion_ops:
            return out
        key = rng.choice(list(out.fusion_ops.keys()))
        current = out.fusion_ops[key]
        candidates = [op for op in context["fusion_operators"] if op != current]
        if candidates:
            out.fusion_ops[key] = rng.choice(candidates)
        return out


class MutateRepresentationIndex(MutationOperator):
    """
    Keep modality set fixed, change repr_idx for one modality leaf.
    """

    name = "mutate_representation_index"

    def __call__(
        self, individual: Individual, rng: random.Random, context: Dict[str, Any]
    ) -> Individual:
        out = copy.deepcopy(individual)
        if not out.leaves:
            return out
        i = rng.randrange(len(out.leaves))
        modality_id, cur_idx = out.leaves[i]
        task_name = context["task_name"]
        reps = context["k_best_representations"][task_name][modality_id]
        if len(reps) <= 1:
            return out
        new_idx = rng.randrange(len(reps))
        while new_idx == cur_idx:
            new_idx = rng.randrange(len(reps))
        out.leaves[i] = (modality_id, new_idx)
        return out


class MutateTreeRotation(MutationOperator):
    """
    Local reassociation (e.g., ((a,b),c) <-> (a,(b,c))) on a random eligible subtree.
    """

    name = "mutate_tree_rotation"

    def __call__(
        self, individual: Individual, rng: random.Random, context: Dict[str, Any]
    ) -> Individual:
        out = copy.deepcopy(individual)
        candidates: List[Tuple[str, str]] = []
        for path in _collect_internal_paths(out.tree):
            subtree = _get_subtree(out.tree, path)
            if isinstance(subtree, int):
                continue
            left, right = subtree
            if not isinstance(left, int):
                candidates.append((path, "left"))
            if not isinstance(right, int):
                candidates.append((path, "right"))

        if not candidates:
            return out

        path, direction = rng.choice(candidates)
        subtree = _get_subtree(out.tree, path)
        left, right = subtree

        if direction == "left":
            left_left, left_right = left
            rotated = (left_left, (left_right, right))
        else:
            right_left, right_right = right
            rotated = ((left, right_left), right_right)

        out.tree = _replace_subtree(out.tree, path, rotated)
        out.fusion_ops = _rebuild_fusion_ops(
            out.tree,
            out.fusion_ops,
            rng,
            context["fusion_operators"],
            randomized_prefixes=[path],
        )
        return out


class MutateSubtreeResample(MutationOperator):
    """
    Pick a random internal subtree, keep the same leaves, but resample:
    - subtree topology
    - fusion operators inside subtree
    """

    name = "mutate_subtree_resample"

    def __call__(
        self, individual: Individual, rng: random.Random, context: Dict[str, Any]
    ) -> Individual:
        out = copy.deepcopy(individual)
        internal_paths = _collect_internal_paths(out.tree)
        if not internal_paths:
            return out

        path = rng.choice(internal_paths)
        subtree = _get_subtree(out.tree, path)
        leaf_indices = _collect_leaf_indices(subtree)
        if len(leaf_indices) < 2:
            return out

        new_subtree = _sample_random_binary_tree_from_leaves(leaf_indices, rng)
        out.tree = _replace_subtree(out.tree, path, new_subtree)
        out.fusion_ops = _rebuild_fusion_ops(
            out.tree,
            out.fusion_ops,
            rng,
            context["fusion_operators"],
            randomized_prefixes=[path],
        )
        return out


class MutateAddModality(MutationOperator):
    """
    Optional global mutation:
    Add one new modality (if below max_modalities), choose repr_idx, fuse with current root.
    """

    name = "mutate_add_modality"

    def __call__(
        self, individual: Individual, rng: random.Random, context: Dict[str, Any]
    ) -> Individual:
        out = copy.deepcopy(individual)
        if len(out.leaves) >= context["max_modalities"]:
            return out

        existing = {m for m, _ in out.leaves}
        candidates = [m for m in context["all_modalities"] if m not in existing]
        if not candidates:
            return out

        m = rng.choice(candidates)
        task_name = context["task_name"]
        reps = context["k_best_representations"][task_name][m]
        if len(reps) == 0:
            return out

        repr_idx = rng.randrange(len(reps))
        new_leaf_idx = len(out.leaves)
        out.leaves.append((m, repr_idx))

        # Wrap old tree with new root fusion
        old_tree = out.tree
        old_fusion_ops = copy.deepcopy(out.fusion_ops)
        out.tree = (old_tree, new_leaf_idx)
        out.fusion_ops = {"": rng.choice(context["fusion_operators"])}
        for path, op in old_fusion_ops.items():
            out.fusion_ops["L" + path] = op
        return out


class MutateDropModality(MutationOperator):
    """
    Optional global mutation:
    Remove one modality leaf (if above min_modalities) and collapse tree.
    """

    name = "mutate_drop_modality"

    def __call__(
        self, individual: Individual, rng: random.Random, context: Dict[str, Any]
    ) -> Individual:
        out = copy.deepcopy(individual)
        if len(out.leaves) <= context["min_modalities"]:
            return out

        leaf_idx = rng.randrange(len(out.leaves))
        new_tree = _remove_leaf_from_tree(out.tree, leaf_idx)
        if new_tree is None:
            return out

        remaining_leaves = [leaf for i, leaf in enumerate(out.leaves) if i != leaf_idx]
        index_map = {}
        next_idx = 0
        for old_idx in range(len(out.leaves)):
            if old_idx == leaf_idx:
                continue
            index_map[old_idx] = next_idx
            next_idx += 1

        out.leaves = remaining_leaves
        out.tree = _reindex_tree(new_tree, index_map)
        out.fusion_ops = _rebuild_fusion_ops(
            out.tree,
            out.fusion_ops,
            rng,
            context["fusion_operators"],
        )
        return out
