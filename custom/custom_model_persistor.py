# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import json

from nvflare.app_common.pt.pt_file_model_persistor import PTFileModelPersistor
from mae.models_custom import build_model
from utils.parse_arg import parse_args

class CustomModelPersistor(PTFileModelPersistor):
    def __init__(
        self,
        model_args=None,
    ):
        super().__init__()
        args = parse_args(model_args)
        self.model = build_model(args)