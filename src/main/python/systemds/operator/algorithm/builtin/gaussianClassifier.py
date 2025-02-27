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
# Autogenerated From : scripts/builtin/gaussianClassifier.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def gaussianClassifier(D: Matrix,
                       C: Matrix,
                       **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     Computes the parameters needed for Gaussian Classification.
     Thus it computes the following per class: the prior probability,
     the inverse covariance matrix, the mean per feature and the determinant
     of the covariance matrix. Furthermore (if not explicitly defined), it
     adds some small smoothing value along the variances, to prevent
     numerical errors / instabilities.
    
    
    
    :param D: Input matrix (training set)
    :param C: Target vector
    :param varSmoothing: Smoothing factor for variances
    :param verbose: Print accuracy of the training set
    :return: Vector storing the class prior probabilities
    :return: Matrix storing the means of the classes
    :return: List of inverse covariance matrices
    :return: Vector storing the determinants of the classes
    """

    params_dict = {'D': D, 'C': C}
    params_dict.update(kwargs)
    
    vX_0 = Matrix(D.sds_context, '')
    vX_1 = Matrix(D.sds_context, '')
    vX_2 = List(D.sds_context, '')
    vX_3 = Matrix(D.sds_context, '')
    output_nodes = [vX_0, vX_1, vX_2, vX_3, ]

    op = MultiReturn(D.sds_context, 'gaussianClassifier', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]
    vX_2._unnamed_input_nodes = [op]
    vX_3._unnamed_input_nodes = [op]

    return op
