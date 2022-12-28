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


import copy
import os
import pprint

import numpy as np
import torch
import torch.optim as optim

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss

import time
from torch.utils.tensorboard import SummaryWriter

import mae.util.misc as misc
from mae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from mae.util.datasets import build_dataset
from mae.models_custom import build_model
from mae.engine_finetune import train_one_epoch, evaluate

from utils.parse_arg import parse_args

class CustomTrainer(Executor):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        aggregation_epochs: int = 1, # 每 round 跑幾個 epoch 後 aggregate
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        fedproxloss_mu: float = 0.0,
        args: str = '',
    ):
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
                
        
        self.dataset_root = dataset_root
        self.aggregation_epochs = aggregation_epochs        
        self.train_task_name = train_task_name        
        self.fedproxloss_mu = fedproxloss_mu
        self.submit_model_task_name = submit_model_task_name        
        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0        
        self.args = parse_args(args)
        self.best_metric_value = 0
        
    def _initialize_trainer(self, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )
        
        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)
        
        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")
        self.device = torch.device(self.args.device)
        self.model = build_model(self.args)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_scaler =  NativeScaler()
        
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
        
        log_dir = os.path.join(self.app_root, self.args.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=log_dir)
        self.mixup_fn = None #TODO
        
    def _terminate_trainer(self):
        # collect threads, close files here
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # the start and end of a run - only happen once
        if event_type == EventType.START_RUN:
            try:
                self._initialize_trainer(fl_ctx)
            except BaseException as e:
                error_msg = f"Exception in _initialize_trainer: {e}"
                self.log_exception(fl_ctx, error_msg)
                self.system_panic(error_msg, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._terminate_trainer()

    def local_train(self, fl_ctx, model_global, abort_signal: Signal, current_round=0):
        for epoch in range(self.aggregation_epochs):
            self.model.train()
            train_stats = train_one_epoch(
                self.model, self.criterion, self.data_loader_train,
                self.optimizer, self.device, epoch, self.loss_scaler,
                self.args.clip_grad, self.mixup_fn,
                log_writer=self.log_writer,
                args=self.args, 
                abort_signal=abort_signal,
                app_root=self.app_root
            )

    def local_eval(self, current_round, abort_signal):
        cur_epoch = current_round
        test_stats = evaluate(self.data_loader_val, self.model, self.device, self.args, 
                              log_writer=self.log_writer, abort_signal=abort_signal, app_root=self.app_root)
        if test_stats is None:
            return None
        
        cur_metric_value = test_stats['auc']
        return cur_metric_value
    
    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({f'best_auc': self.best_metric_value})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def _train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
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
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.data_loader_train)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss
        if self.fedproxloss_mu > 0:
            model_global = copy.deepcopy(self.model)
            for param in model_global.parameters():
                param.requires_grad = False
        else:
            model_global = None

        # perform valid before local train
        global_metric = self.local_eval(current_round, abort_signal)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_metric_global_model: {global_metric:.4f}")

        # local train
        self.local_train(
            fl_ctx=fl_ctx,            
            model_global=model_global,
            abort_signal=abort_signal,
            current_round=current_round,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        val_metric = self.local_eval(current_round, abort_signal)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_metric_local_model: {val_metric:.4f}")

        # save model
        self.save_model(is_best=False)
        if val_metric > self.best_metric_value:
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)
        dxo.set_meta_prop(MetaKey.INITIAL_METRICS, global_metric)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def _submit_model(
        self,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Retrieve the local model saved during training.
        model_data = None
        try:
            # load model to cpu to make it serializable
            model_data = torch.load(self.best_local_model_file, map_location="cpu")
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load best model: {e}")

        # Checking abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Create DXO and shareable from model data.
        if model_data:
            dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_data["model_weights"])
            return dxo.to_shareable()
        else:
            # Set return code.
            self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        self.log_info(fl_ctx, f"Task name: {task_name}")
        if task_name == self.train_task_name:
            return self._train(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
        elif task_name == self.submit_model_task_name:
            return self._submit_model(fl_ctx=fl_ctx, abort_signal=abort_signal)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
