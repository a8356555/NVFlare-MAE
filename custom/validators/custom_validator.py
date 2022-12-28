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

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from mae.util.datasets import build_dataset
from mae.models_custom import build_model
from mae.engine_finetune import evaluate
from utils.parse_arg import parse_args

class CustomValidator(Executor):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        validate_task_name=AppConstants.TASK_VALIDATION,                                                       
        args: str = '',
    ):
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
                
        
        self.dataset_root = dataset_root
        self._validate_task_name = validate_task_name
        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0        
        self.args = parse_args(args)


    def _initialize_validator(self, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # Epoch counter
        self.epoch_of_start_time = 0

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )
        
        self.device = torch.device(self.args.device)
        self.model = build_model(self.args)
        
        dataset_train = build_dataset(is_train=True, args=self.args)
        dataset_val = build_dataset(is_train=False, args=self.args)
        # Create the video train and val loaders.
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        self.data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

        self.data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False
        )

    def _terminate_executor(self):
        # collect threads, close files here
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # the start and end of a run - only happen once
        if event_type == EventType.START_RUN:
            try:
                self._initialize_validator(fl_ctx)
            except BaseException as e:
                error_msg = f"Exception in _initialize_validator: {e}"
                self.log_exception(fl_ctx, error_msg)
                self.system_panic(error_msg, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._terminate_executor()

    def local_valid(self, abort_signal: Signal):
        self.model.eval()
        test_stats = evaluate(self.data_loader_val, self.model, self.device, self.args, 
                              abort_signal=abort_signal, app_root=self.app_root)
        if test_stats is None:
            return None
        
        cur_metric_value = test_stats['auc']
        return cur_metric_value

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == self._validate_task_name:
            # Check abort signal
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # get round information
            self.log_info(fl_ctx, f"Task name: {task_name}")
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

            # update local model weights with received weights
            dxo = from_shareable(shareable)
            global_weights = dxo.data

            # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
            local_var_dict = self.model.state_dict()
            model_keys = global_weights.keys()
            for var_name in local_var_dict:
                if var_name in model_keys:
                    weights = global_weights[var_name]
                    try:
                        # update the local dict
                        local_var_dict[var_name] = torch.as_tensor(np.reshape(weights, local_var_dict[var_name].shape))
                    except Exception as e:
                        raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
            self.model.load_state_dict(local_var_dict)

            # perform valid
            train_metric = self.local_valid(abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"train_metric: {train_metric:.4f}")

            val_metric = self.local_valid(abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric: {val_metric:.4f}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_metric": train_metric, "val_metric": val_metric}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
