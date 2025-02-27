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
# Autogenerated From : scripts/builtin/imputeByMedianApply.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def imputeByMedianApply(X: Matrix,
                        imputedVec: Matrix):
    """
     impute the data by median value and if the feature is categorical then by mode value
     Related to [SYSTEMDS-2662] dependency function for cleaning pipelines
    
    
    
    :param X: Data Matrix (Recoded Matrix for categorical features)
    :param imputationVector: column median vector
    :return: imputed dataset
    """

    params_dict = {'X': X, 'imputedVec': imputedVec}
    return Matrix(X.sds_context,
        'imputeByMedianApply',
        named_input_nodes=params_dict)
