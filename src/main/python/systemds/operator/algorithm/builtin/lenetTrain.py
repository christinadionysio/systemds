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
# Autogenerated From : scripts/builtin/lenetTrain.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def lenetTrain(X: Matrix,
               Y: Matrix,
               X_val: Matrix,
               Y_val: Matrix,
               C: int,
               Hin: int,
               Win: int,
               **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     This builtin function trains LeNet CNN. The architecture of the
     networks is:conv1 -> relu1 -> pool1 -> conv2 -> relu2 -> pool2 ->
     affine3 -> relu3 -> affine4 -> softmax
    
    
    
    :param X: Input data matrix, of shape (N, C*Hin*Win)
    :param Y: Target matrix, of shape (N, K)
    :param X_val: Validation data matrix, of shape (N, C*Hin*Win)
    :param Y_val: Validation target matrix, of shape (N, K)
    :param C: Number of input channels (dimensionality of input depth)
    :param Hin: Input width
    :param Win: Input height
    :param batch_size: Batch size
    :param epochs: Number of epochs
    :param lr: Learning rate
    :param mu: Momentum value
    :param decay: Learning rate decay
    :param reg: Regularization strength
    :param seed: Seed for model initialization
    :param verbose: Flag indicates if function should print to stdout
    :return: Trained model which can be used in lenetPredict
    """

    params_dict = {'X': X, 'Y': Y, 'X_val': X_val, 'Y_val': Y_val, 'C': C, 'Hin': Hin, 'Win': Win}
    params_dict.update(kwargs)
    return Matrix(X.sds_context,
        'lenetTrain',
        named_input_nodes=params_dict)
