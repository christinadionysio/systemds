# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

from __future__ import annotations
import copy
import multiprocessing as mp
import os
import pickle
import random
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple
from deap import base, creator, tools
from systemds.scuro.drsearch.multimodal_ga_optimizer import (
    _collect_internal_paths,
    _get_subtree,
    _replace_subtree,
    _rebuild_fusion_ops,
    _remove_leaf_from_tree,
    _reindex_tree,
    _collect_leaf_indices,
)
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDAGBuilder,
    RepresentationDag,
)
from systemds.scuro.drsearch.task import Task
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.utils.schema_helpers import get_shape

Tree = int | Tuple["Tree", "Tree"]


@dataclass
class FusionSearchResult:
    dag: RepresentationDag
    train_score: dict
    val_score: dict
    test_score: dict
    runtime: float = 0.0
    task_time: float = 0.0
    representation_time: float = 0.0
    task_name: str = ""

    # Per-fold score vectors, {metric: [fold0, ...]}. Needed for error bars and
    # for any selection rule that has to tell a consistent winner apart from a
    # lucky-fold one; the averaged dicts above cannot.
    val_fold_scores: dict = field(default_factory=dict)
    train_fold_scores: dict = field(default_factory=dict)
    test_fold_scores: dict = field(default_factory=dict)

    # Fit/inference wall-clock from the fold loop, per fold and averaged.
    task_timing: dict = field(default_factory=dict)

    # Search trajectory: generation, position in the evaluation sequence, and
    # seconds since this GA run began. Results are appended in evaluation
    # order, but nothing recorded *when* -- which is what a best-so-far curve
    # against a wall-clock budget needs.
    generation: int = -1
    eval_index: int = -1
    t_since_search_start_s: float = 0.0
    t_eval_end_unix: float = 0.0


@dataclass
class DagGenome:
    leaves: List[Tuple[str, int]]
    tree: Tree
    fusion_ops: Dict[str, type]


# ----------------------------------------------------------------------
# Module-level evaluation helpers.
#
# These are intentionally free functions (not bound methods) so that
# `_evaluate_dag_worker` can be shipped to a separate process and pickled
# safely. `_evaluate_genome_body` holds the actual logic and is shared by
# both the in-process (serial) and cross-process (parallel) evaluation
# paths, so the two can never silently diverge in behavior.
# ----------------------------------------------------------------------


# Objectives that come from evaluation timing rather than from the task's
# score dict (scores[i].average_scores). Anything not listed here is looked
# up as a key in the validation score dict (e.g. "accuracy", "f1").
_TIMING_OBJECTIVES = {"runtime", "task_time", "representation_time"}

ObjectiveSpec = Tuple[str, str]  # (name, "max" | "min")


def _objective_value(
    name: str, val_score: Dict[str, float], timing: Dict[str, float]
) -> float:
    if name in _TIMING_OBJECTIVES:
        return timing[name]
    return val_score[name]


def _failure_fitness(objective_specs: List[ObjectiveSpec]) -> Tuple[float, ...]:
    """
    The worst possible value for each objective, direction-aware: -inf for a
    "max" objective, +inf for a "min" objective (e.g. runtime), so that a
    failed evaluation always ranks last regardless of DEAP's per-objective
    fitness weight sign (wvalue = raw_value * weight).
    """
    return tuple(
        float("-inf") if direction == "max" else float("inf")
        for _, direction in objective_specs
    )


def _evaluate_genome_body(
    dag: RepresentationDag,
    task: Task,
    modalities: List[Any],
    objective_specs: List[ObjectiveSpec],
) -> Tuple[Optional[Tuple[float, ...]], Optional[Dict[str, Any]]]:
    start = time.time()
    fused = dag.execute(modalities, task, enable_cache=False)
    if fused is None:
        return None, None

    if isinstance(fused, dict):
        fused = fused[list(fused.keys())[-1]]

    if task.expected_dim == 1 and get_shape(fused.metadata) > 1:
        fused = AggregatedRepresentation().transform(fused)

    t0 = time.time()
    scores = task.run(fused.data)
    task_time = time.time() - t0
    total = time.time() - start

    val_score = scores[1].average_scores
    timing = {
        "runtime": total,
        "task_time": task_time,
        "representation_time": total - task_time,
    }
    fitness = tuple(
        _objective_value(name, val_score, timing) for name, _ in objective_specs
    )
    payload = {
        "train_score": scores[0].average_scores,
        "val_score": val_score,
        "test_score": scores[2].average_scores,
        "train_fold_scores": scores[0].fold_scores(),
        "val_fold_scores": scores[1].fold_scores(),
        "test_fold_scores": scores[2].fold_scores(),
        "task_timing": getattr(task, "last_run_timing", {}),
        **timing,
    }
    return fitness, payload


def _evaluate_dag_worker(
    dag_bytes: bytes,
    task_bytes: bytes,
    modalities_bytes: bytes,
    objective_specs: List[ObjectiveSpec],
) -> Tuple[Tuple[float, ...], Optional[Dict[str, Any]], Optional[str]]:
    """
    Runs in a worker process. Never raises: any failure (a shape-incompatible
    fusion, an OOM, a bug in a representation) is caught and reported back as
    plain, picklable data, so a single bad genome can never take down the
    rest of the population's evaluation.
    """
    try:
        dag = pickle.loads(dag_bytes)
        task = pickle.loads(task_bytes)
        modalities = pickle.loads(modalities_bytes)
        fitness, payload = _evaluate_genome_body(dag, task, modalities, objective_specs)
        if fitness is None:
            fitness = _failure_fitness(objective_specs)
        return fitness, payload, None
    except Exception:
        return _failure_fitness(objective_specs), None, traceback.format_exc()


class MultimodalDeapOptimizer:
    """
    Multimodal optimizer that uses DEAP's genome/fitness containers and
    selection operators to run a genetic search over fusion DAGs.

    @param modalities: List of modalities to optimize.
    @param unimodal_optimization_results: Unimodal optimization results.
    @param tasks: List of tasks to optimize.
    @param debug: Whether to print debug information.
    @param min_modalities: Minimum number of modalities to use.
    @param max_modalities: Maximum number of modalities to use.
    @param metric: Metric to optimize (used when objectives is not given).
    @param objectives: Optional list of (name, direction) pairs to run a
        multi-objective (Pareto/NSGA-II) search instead of optimizing a
        single scalar metric, e.g. [("accuracy", "max"), ("runtime",
        "min")]. "runtime", "task_time", "representation_time" are read
        from evaluation timing; any other name is looked up in the task's
        validation-score dict (e.g. "accuracy", "f1"). When given, this
        overrides metric/maximize_metric. Defaults to None (single
        objective, unchanged behavior).
    @param population_size: Population size.
    @param generations: Maximum number of generations.
    @param crossover_probability: Crossover probability.
    @param mutation_probability: Mutation probability.
    @param random_seed: Random seed.
    @param elite_size: Number of top individuals carried over unchanged
        (with their fitness preserved) into the next generation.
    @param max_workers: Number of worker processes used to evaluate a
        generation's population. 1 (default) evaluates serially in-process.
    @param batch_size: Maximum number of genome evaluations kept in flight
        at once when max_workers > 1. Bounds peak memory/GPU usage
        independently of max_workers; defaults to max_workers.
    @param early_stopping_patience: Stop once this many consecutive
        generations pass without a validation-metric improvement larger
        than early_stopping_min_delta. None disables early stopping.
    @param early_stopping_min_delta: Minimum improvement to reset the
        early-stopping counter.
    @param allow_repeated_modalities: Allow one modality to contribute more
        than one leaf to a genome, each with a different representation
        (intra-modal fusion). Off by default, which keeps the historical
        behaviour of at most one leaf per modality and caps the leaf count at
        len(modalities). Turning it on raises that cap to the total number of
        distinct (modality, representation) leaves available.
    @param hall_of_fame_size: How many incumbents to keep per task in
        single-objective mode (ignored in multi-objective mode, where the
        whole non-dominated front is kept). See get_hall_of_fame.
    @param novelty_breeding: Reject offspring whose genome was already
        evaluated in an *earlier* generation, not just earlier in the
        current one. Without this a converged population refills itself
        with re-treads that hit the fitness cache, cost no wall clock and
        explore nothing -- on the regret-analysis replays 55-84% of all
        fitness requests were such duplicates, so a population of 16
        behaved like a population of ~5. Elites are exempt (they are
        carried over before the archive is consulted), and once the
        archive covers most of the reachable space the retry budget runs
        out and breeding falls back to duplicates, so the search
        degrades gracefully rather than stalling. Defaults to True; set
        False to reproduce pre-fix runs.
    """

    def __init__(
        self,
        modalities: List[Any],
        unimodal_optimization_results: Any,
        tasks: List[Task],
        debug: bool = True,
        min_modalities: int = 2,
        max_modalities: int = None,
        metric: str = "accuracy",
        objectives: Optional[List[ObjectiveSpec]] = None,
        checkpoint_every: int = None,
        resume: bool = True,
        population_size: int = 32,
        generations: int = 20,
        crossover_probability: float = 0.7,
        mutation_probability: float = 0.4,
        random_seed: int = 42,
        maximize_metric: bool = True,
        elite_size: int = 2,
        max_workers: int = 1,
        batch_size: Optional[int] = None,
        early_stopping_patience: Optional[int] = 5,
        early_stopping_min_delta: float = 1e-6,
        novelty_breeding: bool = True,
        hall_of_fame_size: int = 5,
        allow_repeated_modalities: bool = False,
    ):
        self.modalities = modalities
        self.tasks = tasks
        self.debug = debug
        # A genome may hold several leaves of the same modality (different
        # representations of it) only when this is on. With it off, one
        # modality contributes at most one leaf, so the leaf count can never
        # exceed len(modalities) and max_modalities is clamped to that.
        self.allow_repeated_modalities = allow_repeated_modalities

        # min_modalities=1 admits single-leaf (unimodal) genomes. It is not the
        # default -- a "multimodal" search that returns a unimodal pipeline is
        # almost always a bug, which is why the floor used to be a hard 2 --
        # but it is a legitimate configuration for a sweep that asks what the
        # fusion actually buys, so it is now the caller's choice.
        self.min_modalities = max(1, min_modalities)
        requested_max = max_modalities or len(modalities)
        self.max_modalities = (
            requested_max
            if allow_repeated_modalities
            else min(requested_max, len(modalities))
        )
        if self.max_modalities < self.min_modalities:
            raise ValueError(
                f"max_modalities ({self.max_modalities}) is below min_modalities "
                f"({self.min_modalities})"
            )
        self.metric_name = metric
        self.maximize_metric = maximize_metric

        if objectives is not None:
            if len(objectives) < 1:
                raise ValueError(
                    "objectives must contain at least one (name, direction) pair"
                )
            for name, direction in objectives:
                if direction not in ("max", "min"):
                    raise ValueError(
                        f"objective direction must be 'max' or 'min', got "
                        f"{direction!r} for objective {name!r}"
                    )
            self.objective_specs: List[ObjectiveSpec] = list(objectives)
            # Keep metric_name/maximize_metric meaningful for callers that
            # still read them (e.g. _extract_k_best's ranking metric) - must
            # be resolved before _extract_k_best() runs below.
            self.metric_name = self.objective_specs[0][0]
            self.maximize_metric = self.objective_specs[0][1] == "max"
        else:
            self.objective_specs = [
                (self.metric_name, "max" if self.maximize_metric else "min")
            ]
        self.is_multi_objective = len(self.objective_specs) > 1

        if len(self.modalities) < self.min_modalities:
            raise ValueError(
                f"MultimodalDeapOptimizer requires at least {self.min_modalities} "
                f"modalities, got {len(self.modalities)}."
            )

        self.operator_registry = Registry()
        self.fusion_operators = self.operator_registry.get_fusion_operators()
        if not self.fusion_operators:
            raise ValueError(
                "MultimodalDeapOptimizer requires at least one registered "
                "fusion operator."
            )
        self.k_best_representations = self._extract_k_best(
            unimodal_optimization_results
        )

        self.optimization_results: Dict[str, List[FusionSearchResult]] = {}
        self.evaluation_errors: Dict[str, int] = {}

        # Incumbent(s) per task, maintained as evaluations come in, so the
        # winner does not have to be recovered by re-ranking every result the
        # search ever produced. Single-objective: the hall_of_fame_size best
        # by the configured objective. Multi-objective: the (unbounded)
        # non-dominated front over the objective tuples.
        self.hall_of_fame_size = max(1, hall_of_fame_size)
        self.hall_of_fame: Dict[str, List[FusionSearchResult]] = {}
        self._hof_fitness: Dict[str, List[Tuple[float, ...]]] = {}
        self.rng = random.Random(random_seed)
        # Search-trajectory bookkeeping: see FusionSearchResult.eval_index.
        self._eval_counter = 0
        self._current_generation = -1
        self._search_start = time.perf_counter()
        self.population_size = max(1, population_size)
        self.generations = max(1, generations)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.random_seed = random_seed
        self._fitness_cache: Dict[str, Dict[Tuple, Tuple[float, ...]]] = {}

        # Leave room for at least one non-elite offspring per generation,
        # otherwise the next generation would just be a sorted clone of the
        # current one and crossover/mutation would never run. Unused in
        # multi-objective mode (NSGA-II selection is implicitly elitist).
        self.elite_size = max(0, min(elite_size, self.population_size - 1))
        self.max_workers = max(1, max_workers)
        self.batch_size = max(1, batch_size or self.max_workers)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.novelty_breeding = novelty_breeding
        self._current_task_name = None

        desired_weights = tuple(
            1.0 if direction == "max" else -1.0 for _, direction in self.objective_specs
        )
        self._objective_weights = desired_weights
        existing_weights = getattr(
            getattr(creator, "FitnessMax", None), "weights", None
        )
        if existing_weights != desired_weights:
            if hasattr(creator, "Individual"):
                del creator.Individual
            if hasattr(creator, "FitnessMax"):
                del creator.FitnessMax
            creator.create("FitnessMax", base.Fitness, weights=desired_weights)
            creator.create("Individual", list, fitness=creator.FitnessMax)
        elif not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def optimize(
        self,
    ) -> Dict[str, List[FusionSearchResult]]:
        """
        Optimize the multimodal representations for the tasks.
        @return: Dictionary of optimization results for each task.
        """
        for task in self.tasks:
            task_name = task.model.name
            self._current_task_name = task_name
            self.optimization_results.setdefault(task_name, [])
            self.evaluation_errors.setdefault(task_name, 0)

            # Each task restarts the search, so its trajectory restarts too.
            self._eval_counter = 0
            self._search_start = time.perf_counter()

            population = self._build_initial_population(task_name)
            best_ever = None
            no_improve = 0

            for gen in range(self.generations):
                self._current_generation = gen
                self._evaluate_population(population, task)

                if self.is_multi_objective:
                    # No single "best" individual exists with multiple
                    # objectives - track the non-dominated (Pareto) front
                    # instead; progress means that front actually changed.
                    front = tools.sortNondominated(
                        population, len(population), first_front_only=True
                    )[0]
                    front_signature = frozenset(
                        self._genome_signature(ind[0]) for ind in front
                    )
                    if best_ever is None or front_signature != best_ever:
                        best_ever = front_signature
                        no_improve = 0
                    else:
                        no_improve += 1
                    debug_msg = f"front_size={len(front)}"
                else:
                    gen_best = max(population, key=lambda ind: ind.fitness.values[0])
                    if (
                        best_ever is None
                        or gen_best.fitness.values[0]
                        > best_ever.fitness.values[0] + self.early_stopping_min_delta
                    ):
                        best_ever = self._clone_individual(gen_best)
                        no_improve = 0
                    else:
                        no_improve += 1
                    debug_msg = f"best={gen_best.fitness.values[0]:.4f}"

                if self.debug:
                    print(
                        f"[GA] task={task_name} gen={gen} {debug_msg} "
                        f"no_improve={no_improve} "
                        f"errors={self.evaluation_errors.get(task_name, 0)}"
                    )

                stagnated = (
                    self.early_stopping_patience is not None
                    and no_improve >= self.early_stopping_patience
                )
                if stagnated or gen == self.generations - 1:
                    if self.debug and stagnated:
                        print(
                            f"[GA] task={task_name} early stopping after "
                            f"{no_improve} generations without improvement"
                        )
                    break

                population = self._next_generation(population, task_name, task)

        return self.optimization_results

    # ------------------------------------------------------------------
    # Population lifecycle
    # ------------------------------------------------------------------

    def _make_individual(self, genome: DagGenome):
        return creator.Individual([genome])

    def _clone_individual(self, ind):
        clone = creator.Individual([copy.deepcopy(ind[0])])
        if ind.fitness.valid:
            clone.fitness.values = ind.fitness.values
        return clone

    def _append_if_unique(
        self,
        population: List[Any],
        genome: DagGenome,
        seen_signatures: set,
    ) -> bool:
        if len(population) >= self.population_size:
            return False
        sig = self._genome_signature(genome)
        if sig in seen_signatures:
            return False
        seen_signatures.add(sig)
        population.append(self._make_individual(genome))
        return True

    def _build_initial_population(self, task_name: str) -> List[Any]:
        population: List[Any] = []
        seen: set = set()
        retry_budget = max(20, self.population_size * 10)
        retries = 0
        while len(population) < self.population_size and retries < retry_budget:
            genome = self._random_genome(task_name)
            if self._append_if_unique(population, genome, seen):
                retries = 0
            else:
                retries += 1
        # Search space smaller than population_size: fall back to
        # (possibly duplicated) random individuals rather than looping
        # forever.
        while len(population) < self.population_size:
            population.append(self._make_individual(self._random_genome(task_name)))
        return population

    def _next_generation(
        self, population: List[Any], task_name: str, task: Task
    ) -> List[Any]:
        if self.is_multi_objective:
            # Classic (mu+lambda) NSGA-II: breed a full offspring pool,
            # evaluate it, then select the next generation from parents +
            # offspring combined via non-dominated sorting + crowding
            # distance. This is inherently elitist (non-dominated parents
            # survive on their own merit), so there's no separate elite_size
            # concept here.
            offspring = self._breed_offspring(
                population, task_name, seen=self._novelty_archive(task_name)
            )
            self._evaluate_population(offspring, task)
            combined = list(population) + list(offspring)
            return list(tools.selNSGA2(combined, self.population_size))

        ranked = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        elite = [self._clone_individual(ind) for ind in ranked[: self.elite_size]]
        seen = {self._genome_signature(ind[0]) for ind in elite}
        seen |= self._novelty_archive(task_name)
        return self._breed_offspring(population, task_name, initial=elite, seen=seen)

    def _novelty_archive(self, task_name: str) -> set:
        """Signatures already evaluated for this task, or empty if disabled.

        `_breed_offspring` dedups against `seen`, which without this holds
        only the current generation's elite -- so an offspring identical to
        something evaluated three generations ago is accepted, served from
        `_fitness_cache`, and occupies a population slot that explores
        nothing. Feeding the whole cache in makes the dedup global.
        """
        if not self.novelty_breeding:
            return set()
        return set(self._fitness_cache.get(task_name, {}))

    def _breed_offspring(
        self,
        population: List[Any],
        task_name: str,
        initial: Optional[List[Any]] = None,
        seen: Optional[set] = None,
    ) -> List[Any]:
        """
        Fills a new population of size population_size via tournament
        selection + crossover + mutation, starting from `initial` (e.g. an
        elite carryover) if given. Falls back to fresh random immigrants
        (allowing duplicates) once the retry budget is exhausted, so a
        search space smaller than population_size can never hang.
        """
        next_population = list(initial) if initial else []
        seen = set(seen) if seen else set()

        retry_budget = max(20, self.population_size * 10)
        retries = 0
        tournsize = max(1, min(3, len(population)))
        while len(next_population) < self.population_size and retries < retry_budget:
            p1, p2 = tools.selTournament(population, 2, tournsize=tournsize)

            if self.rng.random() < self.crossover_probability:
                g1, g2 = self._crossover_genomes(p1[0], p2[0])
            else:
                g1, g2 = copy.deepcopy(p1[0]), copy.deepcopy(p2[0])

            if self.rng.random() < self.mutation_probability:
                g1 = self._mutate_genome(g1, task_name)
            if self.rng.random() < self.mutation_probability:
                g2 = self._mutate_genome(g2, task_name)

            added1 = self._append_if_unique(next_population, g1, seen)
            added2 = self._append_if_unique(next_population, g2, seen)
            retries = 0 if (added1 or added2) else retries + 1

        # Ran out of retries (tiny search space): top up with fresh random
        # immigrants, allowing duplicates rather than looping forever.
        while len(next_population) < self.population_size:
            genome = self._random_genome(task_name)
            if not self._append_if_unique(next_population, genome, seen):
                next_population.append(self._make_individual(genome))

        return next_population

    # ------------------------------------------------------------------
    # Evaluation (serial + bounded parallel)
    # ------------------------------------------------------------------

    def _evaluate_population(self, population: List[Any], task: Task) -> None:
        to_evaluate = [ind for ind in population if not ind.fitness.valid]
        if not to_evaluate:
            return
        if self.max_workers > 1 and len(to_evaluate) > 1:
            self._evaluate_individuals_parallel(to_evaluate, task)
        else:
            for ind in to_evaluate:
                fitness = self._evaluate_genome(ind[0], task)
                ind.fitness.values = fitness

    def _evaluate_individuals_parallel(
        self, individuals: List[Any], task: Task
    ) -> None:
        task_name = task.model.name
        cache = self._fitness_cache.setdefault(task_name, {})
        ctx = mp.get_context("spawn")
        task_bytes = pickle.dumps(task)
        futures: Dict[Any, Tuple[Any, RepresentationDag, Tuple]] = {}
        pending_followers: Dict[Tuple, List[Any]] = {}

        def _drain(done_futures):
            for fut in done_futures:
                ind, dag, sig = futures.pop(fut)
                fitness, payload, error = fut.result()
                ind.fitness.values = fitness
                self._record_evaluation(task_name, dag, payload, error)
                cache[sig] = fitness
                for follower in pending_followers.pop(sig, []):
                    follower.fitness.values = fitness

        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=ctx
        ) as executor:
            for ind in individuals:
                genome = ind[0]
                sig = self._genome_signature(genome)
                cached = cache.get(sig)
                if cached is not None:
                    ind.fitness.values = cached
                    continue
                if sig in pending_followers:
                    pending_followers[sig].append(ind)
                    continue

                dag = self._genome_to_dag(genome)
                modalities = list(
                    chain.from_iterable(self.k_best_representations[task_name].values())
                )
                fut = executor.submit(
                    _evaluate_dag_worker,
                    pickle.dumps(dag),
                    task_bytes,
                    pickle.dumps(modalities),
                    self.objective_specs,
                )
                futures[fut] = (ind, dag, sig)
                pending_followers[sig] = []

                if len(futures) >= self.batch_size:
                    done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                    _drain(done)

            if futures:
                done, _ = wait(set(futures.keys()))
                _drain(done)

    def _record_evaluation(
        self,
        task_name: str,
        dag: RepresentationDag,
        payload: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        self.optimization_results.setdefault(task_name, [])
        if error is not None or payload is None:
            self.evaluation_errors[task_name] = (
                self.evaluation_errors.get(task_name, 0) + 1
            )
            if self.debug and error is not None:
                last_line = error.strip().splitlines()[-1] if error.strip() else error
                print(
                    f"[GA] genome evaluation failed for task={task_name}: {last_line}"
                )
            return

        result = FusionSearchResult(
            dag=dag,
            train_score=payload["train_score"],
            val_score=payload["val_score"],
            test_score=payload["test_score"],
            train_fold_scores=payload.get("train_fold_scores", {}),
            val_fold_scores=payload.get("val_fold_scores", {}),
            test_fold_scores=payload.get("test_fold_scores", {}),
            task_timing=payload.get("task_timing", {}),
            runtime=payload["runtime"],
            task_time=payload["task_time"],
            representation_time=payload["representation_time"],
            task_name=task_name,
            generation=self._current_generation,
            eval_index=self._eval_counter,
            t_since_search_start_s=time.perf_counter() - self._search_start,
            t_eval_end_unix=time.time(),
        )
        self.optimization_results.setdefault(task_name, []).append(result)
        self._update_hall_of_fame(task_name, result)
        self._eval_counter += 1

    # ------------------------------------------------------------------
    # Hall of fame
    # ------------------------------------------------------------------

    def _dominates(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        """True if objective tuple `a` Pareto-dominates `b`, direction-aware."""
        wa = [w * v for w, v in zip(self._objective_weights, a)]
        wb = [w * v for w, v in zip(self._objective_weights, b)]
        return all(x >= y for x, y in zip(wa, wb)) and any(
            x > y for x, y in zip(wa, wb)
        )

    def _update_hall_of_fame(self, task_name: str, result: FusionSearchResult) -> None:
        timing = {
            "runtime": result.runtime,
            "task_time": result.task_time,
            "representation_time": result.representation_time,
        }
        try:
            fitness = tuple(
                _objective_value(name, result.val_score, timing)
                for name, _ in self.objective_specs
            )
        except KeyError:
            # An objective the task did not report for this candidate: it
            # cannot be ranked, so it simply does not enter the hall of fame.
            return

        hof = self.hall_of_fame.setdefault(task_name, [])
        fits = self._hof_fitness.setdefault(task_name, [])

        if self.is_multi_objective:
            if any(self._dominates(f, fitness) or f == fitness for f in fits):
                return
            keep = [i for i, f in enumerate(fits) if not self._dominates(fitness, f)]
            self.hall_of_fame[task_name] = [hof[i] for i in keep] + [result]
            self._hof_fitness[task_name] = [fits[i] for i in keep] + [fitness]
            return

        weight = self._objective_weights[0]
        hof.append(result)
        fits.append(fitness)
        order = sorted(
            range(len(fits)), key=lambda i: weight * fits[i][0], reverse=True
        )
        order = order[: self.hall_of_fame_size]
        self.hall_of_fame[task_name] = [hof[i] for i in order]
        self._hof_fitness[task_name] = [fits[i] for i in order]

    def get_hall_of_fame(self, task_name: str) -> List[FusionSearchResult]:
        """The incumbent(s) for a task: the best `hall_of_fame_size` results by
        the configured objective, or the non-dominated front in multi-objective
        mode. Empty if nothing evaluated successfully."""
        return list(self.hall_of_fame.get(task_name, []))

    def _extract_k_best(self, unimodal_results) -> Dict[str, Dict[str, List[Any]]]:
        """
        Extract the k best representations for each modality and task.
        @param unimodal_results: Unimodal optimization results.
        @return: Dictionary of k best representations for each modality and task.
        """
        k_best = {}
        for task in self.tasks:
            name = task.model.name
            k_best[name] = {}
            for modality in self.modalities:
                _, cached_data = unimodal_results.get_k_best_results(
                    modality, task, self.metric_name
                )
                k_best[name][modality.modality_id] = cached_data
        return k_best

    def _available_modality_ids(self, task_name: str) -> List[Any]:
        """Modalities with at least one representation for this task.

        Datasets like StressID can be missing a modality for some
        tasks/instances, and a leaf pointing at an empty representation list
        is unresolvable.
        """
        reps = self.k_best_representations[task_name]
        return [
            m.modality_id
            for m in self.modalities
            if len(reps.get(m.modality_id, [])) > 0
        ]

    def _leaf_capacity(self, task_name: str) -> int:
        """How many distinct leaves a genome may hold at most.

        One per modality normally; with allow_repeated_modalities, one per
        (modality, representation) pair, since that is what makes a leaf
        distinct.
        """
        reps = self.k_best_representations[task_name]
        ids = self._available_modality_ids(task_name)
        if not self.allow_repeated_modalities:
            return len(ids)
        return sum(len(reps[mid]) for mid in ids)

    def _random_genome(self, task_name: str) -> DagGenome:
        """
        Generate a random genome for a given task.
        @param task_name: Name of the task.
        @return: Random genome.
        """
        reps = self.k_best_representations[task_name]
        available_modality_ids = self._available_modality_ids(task_name)
        capacity = self._leaf_capacity(task_name)
        if capacity < self.min_modalities:
            raise ValueError(
                f"Need at least {self.min_modalities} distinct leaves for task "
                f"'{task_name}', but only {capacity} are available across "
                f"{len(available_modality_ids)} modalities."
            )

        upper = min(self.max_modalities, capacity)
        lower = min(self.min_modalities, upper)
        r = self.rng.randint(lower, upper)

        if self.allow_repeated_modalities:
            # Sample distinct (modality, representation) leaves; a modality may
            # appear more than once as long as it contributes a different
            # representation each time.
            pool = [
                (mid, idx)
                for mid in available_modality_ids
                for idx in range(len(reps[mid]))
            ]
            leaves = self.rng.sample(pool, r)
        else:
            chosen = self.rng.sample(available_modality_ids, r)
            leaves = [(mid, self.rng.randrange(len(reps[mid]))) for mid in chosen]

        tree = self._random_binary_tree(len(leaves))
        fusion_ops = {}
        self._assign_fusion_ops(tree, fusion_ops, "")
        return DagGenome(leaves=leaves, tree=tree, fusion_ops=fusion_ops)

    def _internal_paths(self, tree):
        return _collect_internal_paths(tree)

    def _random_binary_tree(self, n: int) -> Tree:
        """
        Generate a random binary tree with n leaves.
        @param n: Number of leaves.
        @return: Random binary tree.
        """
        nodes: List[Tree] = list(range(n))
        while len(nodes) > 1:
            i, j = self.rng.sample(range(len(nodes)), 2)
            a, b = nodes.pop(max(i, j)), nodes.pop(min(i, j))
            nodes.append((a, b))
        return nodes[0]

    def _assign_fusion_ops(self, subtree: Tree, ops: Dict[str, Any], path: str) -> None:
        """
        Assign fusion operators to the internal nodes of a binary tree.
        @param subtree: Binary tree.
        @param ops: Dictionary of fusion operators.
        @param path: Path to the current node.
        """
        if isinstance(subtree, int):
            return
        ops[path] = self.rng.choice(self.fusion_operators)
        left, right = subtree
        self._assign_fusion_ops(left, ops, path + "L")
        self._assign_fusion_ops(right, ops, path + "R")

    def _genome_to_dag(self, genome: DagGenome) -> RepresentationDag:
        """
        Convert a genome to a DAG.
        @param genome: Genome.
        @return: RepresentationDAG.
        """
        builder = RepresentationDAGBuilder()
        leaf_ids = [
            builder.create_leaf_node(mod_id, repr_idx)
            for mod_id, repr_idx in genome.leaves
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

    def _genome_signature(self, g: DagGenome) -> Tuple:
        """
        Generate a signature for a genome to be used as a cache key.
        @param g: Genome.
        @return: Signature.
        """

        def norm(t: Tree):
            return t if isinstance(t, int) else (norm(t[0]), norm(t[1]))

        return (
            tuple(g.leaves),
            norm(g.tree),
            tuple(sorted((p, c.__name__) for p, c in g.fusion_ops.items())),
        )

    def _evaluate_genome(self, genome: DagGenome, task: Task) -> Tuple[float, ...]:
        """
        Evaluate a genome for a given task (in-process). Stores the result
        in the optimization results, or records a failure without raising
        if the DAG/task fails to execute (e.g. an incompatible fusion).
        @param genome: Genome.
        @param task: Task.
        @return: Fitness tuple, one value per configured objective.
        """
        task_name = task.model.name
        sig = self._genome_signature(genome)
        cache = self._fitness_cache.setdefault(task_name, {})
        if sig in cache:
            return cache[sig]

        dag = self._genome_to_dag(genome)
        modalities = list(
            chain.from_iterable(self.k_best_representations[task_name].values())
        )

        try:
            fitness, payload = _evaluate_genome_body(
                dag, task, modalities, self.objective_specs
            )
            error = None
            if fitness is None:
                fitness = _failure_fitness(self.objective_specs)
        except Exception:
            fitness = _failure_fitness(self.objective_specs)
            payload, error = None, traceback.format_exc()

        self._record_evaluation(task_name, dag, payload, error)
        cache[sig] = fitness
        return fitness

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover_genomes(
        self, g1: DagGenome, g2: DagGenome
    ) -> Tuple[DagGenome, DagGenome]:
        """
        Recombine two genomes.

        When both parents agree on the exact same modality/representation
        leaves, a classic reciprocal subtree swap is performed. Otherwise
        (the common case, since leaves are sampled independently per
        genome) subtree crossover over leaf indices would be meaningless,
        so fusion-operator choices are mixed at whichever internal-path
        keys the two trees happen to share instead - every non-trivial
        tree has at least the root path "" in common, so this always
        performs real recombination rather than silently returning the
        parents unchanged.
        @param g1: First parent genome.
        @param g2: Second parent genome.
        @return: Two child genomes.
        """
        c1, c2 = copy.deepcopy(g1), copy.deepcopy(g2)

        if c1.leaves == c2.leaves:
            paths1 = self._internal_paths(c1.tree)
            paths2 = self._internal_paths(c2.tree)
            if paths1 and paths2:
                path1 = self.rng.choice(paths1)
                path2 = self.rng.choice(paths2)
                subtree1 = _get_subtree(c1.tree, path1)
                subtree2 = _get_subtree(c2.tree, path2)
                c1.tree = _replace_subtree(c1.tree, path1, subtree2)
                c2.tree = _replace_subtree(c2.tree, path2, subtree1)
                c1.fusion_ops = _rebuild_fusion_ops(
                    c1.tree,
                    {**c1.fusion_ops, **c2.fusion_ops},
                    self.rng,
                    self.fusion_operators,
                    randomized_prefixes=[path1],
                )
                c2.fusion_ops = _rebuild_fusion_ops(
                    c2.tree,
                    {**c2.fusion_ops, **c1.fusion_ops},
                    self.rng,
                    self.fusion_operators,
                    randomized_prefixes=[path2],
                )
            return c1, c2

        shared_paths = set(c1.fusion_ops) & set(c2.fusion_ops)
        for path in shared_paths:
            if self.rng.random() < 0.5:
                c1.fusion_ops[path], c2.fusion_ops[path] = (
                    c2.fusion_ops[path],
                    c1.fusion_ops[path],
                )
        return c1, c2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate_genome(self, g: DagGenome, task_name: str) -> DagGenome:
        op = self.rng.choice(
            [
                self._mutate_change_fusion,
                lambda gg: self._mutate_swap_leaf_repr(gg, task_name),
                lambda gg: self.mutate_add_leaf(gg, task_name),
                self.mutate_remove_leaf,
                self.mutate_replace_subtree,
            ]
        )
        return op(g)

    def _mutate_change_fusion(self, g: DagGenome) -> DagGenome:
        """
        Change a fusion operator at a random internal node.
        @param g: Genome.
        @return: Mutated genome.
        """
        g = copy.deepcopy(g)
        paths = [p for p in g.fusion_ops]
        if not paths:
            return g
        path = self.rng.choice(paths)
        choices = [op for op in self.fusion_operators if op != g.fusion_ops[path]]
        if choices:
            g.fusion_ops[path] = self.rng.choice(choices)
        return g

    def _mutate_swap_leaf_repr(self, g: DagGenome, task_name: str) -> DagGenome:
        """
        Change which k-best unimodal repr a leaf uses (same modality).
        @param g: Genome.
        @param task_name: Name of the task.
        @return: Mutated genome.
        """
        g = copy.deepcopy(g)
        i = self.rng.randrange(len(g.leaves))
        mod_id, current = g.leaves[i]
        k = len(self.k_best_representations[task_name][mod_id])
        if k <= 1:
            return g
        # With repeated modalities allowed the genome may already hold another
        # leaf of this modality; swapping onto that representation would make
        # the two leaves identical, which wastes a slot on a duplicate.
        taken = {
            idx for j, (mid, idx) in enumerate(g.leaves) if mid == mod_id and j != i
        }
        choices = [idx for idx in range(k) if idx != current and idx not in taken]
        if not choices:
            return g
        g.leaves[i] = (mod_id, self.rng.choice(choices))
        return g

    def mutate_add_leaf(self, g: DagGenome, task_name: str) -> DagGenome:
        """
        Add a new leaf to the genome (adding a new modality). Only
        modalities that have at least one representation for this task are
        considered.
        @param g: Genome.
        @param task_name: Name of the task.
        @return: Mutated genome.
        """
        if len(g.leaves) >= self.max_modalities:
            return g
        g = copy.deepcopy(g)
        reps = self.k_best_representations[task_name]
        if self.allow_repeated_modalities:
            # Any (modality, representation) leaf the genome does not already
            # hold, so a modality can be added a second time under a different
            # representation.
            existing_leaves = set(g.leaves)
            candidates = [
                (mid, idx)
                for mid in self._available_modality_ids(task_name)
                for idx in range(len(reps[mid]))
                if (mid, idx) not in existing_leaves
            ]
            if not candidates:
                return g
            new_leaf = self.rng.choice(candidates)
        else:
            existing = {l[0] for l in g.leaves}
            available = [
                m.modality_id
                for m in self.modalities
                if m.modality_id not in existing
                and len(reps.get(m.modality_id, [])) > 0
            ]
            if not available:
                return g
            mod_id = self.rng.choice(available)
            new_leaf = (mod_id, self.rng.randrange(len(reps[mod_id])))
        new_idx = len(g.leaves)
        g.leaves.append(new_leaf)
        if isinstance(g.tree, int):
            g.tree = (g.tree, new_idx)
            g.fusion_ops = {"": self.rng.choice(self.fusion_operators)}
        else:
            paths = self._internal_paths(g.tree)
            path = self.rng.choice(paths)
            sub = _get_subtree(g.tree, path)
            g.tree = _replace_subtree(g.tree, path, (sub, new_idx))
            # `sub` (and everything under it) just shifted one level deeper
            # in the tree (from `path` to `path + "L"`), so its existing
            # fusion_ops entries are keyed under stale paths and must be
            # rebuilt rather than patched in place - only the brand-new
            # node at `path` needs a freshly chosen operator.
            g.fusion_ops = _rebuild_fusion_ops(
                g.tree,
                g.fusion_ops,
                self.rng,
                self.fusion_operators,
                randomized_prefixes=[path],
            )
        return g

    def mutate_remove_leaf(self, g: DagGenome) -> DagGenome:
        """
        Remove a leaf from the genome (removing a modality).
        @param g: Genome.
        @return: Mutated genome.
        """
        if len(g.leaves) <= self.min_modalities:
            return g
        g = copy.deepcopy(g)
        drop = self.rng.randrange(len(g.leaves))
        new_tree = _remove_leaf_from_tree(g.tree, drop)
        if new_tree is None:
            return g
        # Reindex leaves + fusion_ops paths
        keep = [i for i in range(len(g.leaves)) if i != drop]
        index_map = {old: new for new, old in enumerate(keep)}
        g.leaves = [g.leaves[i] for i in keep]
        g.tree = _reindex_tree(new_tree, index_map)
        g.fusion_ops = _rebuild_fusion_ops(
            g.tree, g.fusion_ops, self.rng, self.fusion_operators
        )
        return g

    def mutate_replace_subtree(self, g: DagGenome) -> DagGenome:
        """
        Replace a subtree with a random subtree.

        The collapse branch (replacing an internal node by one of its
        children) drops every leaf on the other side of that node, so it is
        only taken when at least min_modalities leaves survive -- otherwise a
        search configured min_modalities=2 silently evaluates unimodal
        pipelines. The dropped leaves are also removed from `genome.leaves`
        and the tree reindexed; leaving them in place used to desynchronise
        the genome from its own tree.
        @param g: Genome.
        @return: Mutated genome.
        """
        g = copy.deepcopy(g)
        paths = self._internal_paths(g.tree)
        if not paths:
            return g
        path = self.rng.choice(paths)
        sub = _get_subtree(g.tree, path)
        if isinstance(sub, int):
            return g

        if self.rng.random() < 0.5:
            child = sub[0] if self.rng.random() < 0.5 else sub[1]
            candidate = _replace_subtree(g.tree, path, child)
            kept = sorted(set(_collect_leaf_indices(candidate)))
            if len(kept) >= self.min_modalities:
                index_map = {old: new for new, old in enumerate(kept)}
                g.leaves = [g.leaves[i] for i in kept]
                g.tree = _reindex_tree(candidate, index_map)
                g.fusion_ops = _rebuild_fusion_ops(
                    g.tree, {}, self.rng, self.fusion_operators
                )
                return g
            # Collapsing here would leave fewer leaves than min_modalities,
            # so fall through to the reshuffle branch, which is leaf-preserving.

        leaf_idxs = _collect_leaf_indices(sub)
        new_sub = self._random_binary_tree(len(leaf_idxs))
        local_map = {i: leaf_idxs[i] for i in range(len(leaf_idxs))}
        new_sub = _reindex_tree(new_sub, local_map)
        g.tree = _replace_subtree(g.tree, path, new_sub)
        g.fusion_ops = _rebuild_fusion_ops(
            g.tree,
            g.fusion_ops,
            self.rng,
            self.fusion_operators,
            randomized_prefixes=[path],
        )
        return g

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_results(self, file_name: str = None, overwrite: bool = False) -> str:
        """
        Persist optimization_results to disk.

        Refuses to clobber an existing file unless overwrite=True is
        passed explicitly. The write itself is atomic (temp file +
        os.replace), so a crash mid-write can never leave a corrupted or
        truncated results file behind.
        @param file_name: Destination path. A timestamped name is
            generated if omitted.
        @param overwrite: Set True to explicitly replace an existing file
            at file_name.
        @return: The path the results were written to.
        """
        if file_name is None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"multimodal_optimizer_{timestr}.pkl"

        directory = os.path.dirname(file_name) or "."
        os.makedirs(directory, exist_ok=True)

        if os.path.exists(file_name) and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing results file '{file_name}'. "
                "Pass overwrite=True if this is intentional, or choose a "
                "different file_name."
            )

        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".tmp_multimodal_results_", suffix=".pkl"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.optimization_results, f)
            os.replace(tmp_path, file_name)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return file_name
