# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
from sklearn import metrics

import torch

from timm.data import Mixup
from timm.utils import accuracy
import numpy as np
import os

import mae.util.misc as misc
import mae.util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, abort_signal=None, app_root=''):
    model.train(True)
    model.to(device)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    all_preds = []
    all_labels = []
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if abort_signal.triggered:
            return None
        
        if data_iter_step == 0 and epoch == 0:
            if args.heterogeneous:
                np.save(os.path.join(app_root, 'input_image.npy'), samples[0].cpu().numpy())
                np.save(os.path.join(app_root, 'input_statistic.npy'), samples[1].cpu().numpy())
            else:
                np.save(os.path.join(app_root, 'input.npy'), samples.cpu().numpy())
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if args.heterogeneous:
            samples = [data.to(device, non_blocking=True) for data in samples]
        else:
            samples = samples.to(device, non_blocking=True)
        all_labels.append(targets)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # TODO: amp
        outputs = model(samples)
        all_preds.append(outputs.softmax(axis=1).cpu().detach())
        loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        loss /= accum_iter
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=False,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()    

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 2))

        batch_size = len(targets)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce)
            log_writer.add_scalar('lr', max_lr)
    all_preds = torch.concat(all_preds).numpy()
    all_labels = torch.concat(all_labels).numpy()
    if args.nb_classes == 2:
        auc = metrics.roc_auc_score(all_labels, all_preds[:, 1])
    else:
        auc = metrics.roc_auc_score(all_labels, all_preds, multi_class='ovo')

    metric_logger.meters['auc'].update(auc, n=1)
    if log_writer is not None:
        log_writer.add_scalar('auc', auc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    with open(os.path.join(app_root, 'metric_logs.txt'), 'a') as f:
        f.write('[TRAIN] AUC {auc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}\n'
          .format(auc=metric_logger.auc, top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, 
             log_writer=None, epoch=None, abort_signal=None, app_root=''):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.to(device)
    model.eval()
    all_targets = []
    all_outputs = []
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):    
        if abort_signal.triggered:
            return None
        
        target = batch[-1]
        target = target.to(device, non_blocking=True)
        samples = batch[0]
        if args.heterogeneous:
            samples = [data.to(device, non_blocking=True) for data in samples]
        else:    
            samples = samples.to(device, non_blocking=True)
        
        # compute output
        # TODO: amp
        output = model(samples)
        loss = criterion(output, target)
        if i == 0:
            if args.heterogeneous:
                np.save(os.path.join(app_root, 'input_valid_image.npy'), samples[0].cpu().numpy())
                np.save(os.path.join(app_root, 'input_valid_statistic.npy'), samples[1].cpu().numpy())
            else:
                np.save(os.path.join(app_root, 'input_valid.npy'), samples.cpu().numpy())
        all_targets.append(target.cpu())
        all_outputs.append(output.softmax(axis=1).cpu())
            
        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = len(target)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        accum_iter = args.accum_iter
        if log_writer is not None and (i + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss.item())
    # gather the stats from all processes
    all_outputs = torch.concat(all_outputs).numpy()
    all_targets = torch.concat(all_targets).numpy()
    if args.nb_classes == 2:
        auc = metrics.roc_auc_score(all_targets, all_outputs[:, 1])
    else:
        auc = metrics.roc_auc_score(all_targets, all_outputs, multi_class='ovo')

    if log_writer is not None:
        log_writer.add_scalar('val_auc', auc)
    metric_logger.meters['auc'].update(auc, n=1)
    metric_logger.synchronize_between_processes()
    with open(os.path.join(app_root, 'metric_logs.txt'), 'a') as f:
        f.write('[TEST] AUC {auc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}\n'
          .format(auc=metric_logger.auc, top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    np.save(os.path.join(app_root, 'all_target_valid.npy'), all_targets)
    np.save(os.path.join(app_root, 'all_output_valid.npy'), all_outputs)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}