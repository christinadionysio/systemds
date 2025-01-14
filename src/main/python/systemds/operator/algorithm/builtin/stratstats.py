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

# Autogenerated By   : src/main/python/generator/generator.py
# Autogenerated From : scripts/builtin/stratstats.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def stratstats(X: Matrix,
               **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     The stratstats.dml script computes common bivariate statistics, such as correlation, slope, and their p-value,
     in parallel for many pairs of input variables in the presence of a confounding categorical variable.
    
     Output contains:
     (1st covariante, 2nd covariante)
     40 columns containing the following information:
     Col 01: 1st covariate X-column number
     Col 02: 1st covariate global presence count
     Col 03: 1st covariate global mean
     Col 04: 1st covariate global standard deviation
     Col 05: 1st covariate stratified standard deviation
     Col 06: R-squared, 1st covariate vs. strata
     Col 07: adjusted R-squared, 1st covariate vs. strata
     Col 08: P-value, 1st covariate vs. strata
     Col 09-10: Reserved
     Col 11: 2nd covariate Y-column number
     Col 12: 2nd covariate global presence count
     Col 13: 2nd covariate global mean
     Col 14: 2nd covariate global standard deviation
     Col 15: 2nd covariate stratified standard deviation
     Col 16: R-squared, 2nd covariate vs. strata
     Col 17: adjusted R-squared, 2nd covariate vs. strata
     Col 18: P-value, 2nd covariate vs. strata
     Col 19-20: Reserved
     Col 21: Global 1st & 2nd covariate presence count
     Col 22: Global regression slope (2nd vs. 1st covariate)
     Col 23: Global regression slope standard deviation
     Col 24: Global correlation = +/- sqrt(R-squared)
     Col 25: Global residual standard deviation
     Col 26: Global R-squared
     Col 27: Global adjusted R-squared
     Col 28: Global P-value for hypothesis "slope = 0"
     Col 29-30: Reserved
     Col 31: Stratified 1st & 2nd covariate presence count
     Col 32: Stratified regression slope (2nd vs. 1st covariate)
     Col 33: Stratified regression slope standard deviation
     Col 34: Stratified correlation = +/- sqrt(R-squared)
     Col 35: Stratified residual standard deviation
     Col 36: Stratified R-squared
     Col 37: Stratified adjusted R-squared
     Col 38: Stratified P-value for hypothesis "slope = 0"
     Col 39: Number of strata with at least two counted points
     Col 40: Reserved
    
    
    
    
    :param X: Matrix X that has all 1-st covariates
    :param Y: Matrix Y that has all 2-nd covariates
        the default value empty means "use X in place of Y"
    :param S: Matrix S that has the stratum column
        the default value empty means "use X in place of S"
    :param Xcid: 1-st covariate X-column indices
        the default value empty means "use columns 1 : ncol(X)"
    :param Ycid: 2-nd covariate Y-column indices
        the default value empty means "use columns 1 : ncol(Y)"
    :param Scid: Column index of the stratum column in S
    :return: Output matrix, one row per each distinct pair
    """

    params_dict = {'X': X}
    params_dict.update(kwargs)
    return Matrix(X.sds_context,
        'stratstats',
        named_input_nodes=params_dict)
