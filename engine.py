import datetime
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import math
from timm.optim import create_optimizer
import numpy as np
import torch
from timm.utils import accuracy

import utils

anchor_dict={}
R_dict={}

def dis_loss(reps_dict,Lambda,Alpha,Beta,M):
    d_loss = 0
    for k1, v1 in reps_dict.items():
        pos = 0
        neg = 0
        for k2, v2 in reps_dict.items():
            d = torch.cdist(anchor_dict[k1].unsqueeze(0), v2)
            if k1 == k2:
                pos = torch.sum(torch.exp(Alpha * ( d- R_dict[k1])))
            else:
                neg += torch.sum(torch.exp(-Beta * (d - R_dict[k1] - M)))
        d_loss += Lambda * (R_dict[k1] ** 2) + (1 / Alpha) * math.log(1 + pos) + (
                        1 / Beta) * math.log(1 + neg)
    return torch.tensor(d_loss)


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int , max_norm: float = 0,
                    is_training=True, task_id=-1, class_mask=None, args = None,):
    model.train(is_training)
    original_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    e_reps_dict = {}
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        b_reps_dict={}
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = original_model(input)
            cls_features = output['pre_logits']
        if task_id==0:
            prompts_matrix = torch.zeros(input.shape[0], args.pool_size*args.base_ways).to(device, non_blocking=True)
            for i, c in enumerate(class_mask[task_id]):
                for j, t in enumerate(target):
                        if t == c:
                            p = i * args.pool_size
                            l = i * args.pool_size + args.pool_size
                            prompts_matrix[j][p:l] = 1
        else:
            prompts_matrix = torch.zeros(input.shape[0], args.pool_size * args.ways).to(device, non_blocking=True)
            for i, c in enumerate(class_mask[task_id]):
                for j, t in enumerate(target):
                    if t == c:
                        p = i * args.pool_size
                        l = i * args.pool_size + args.pool_size
                        prompts_matrix[j][p:l] = 1
        # prompts_matrix = torch.zeros(input.shape[0], args.pool_size).to(device, non_blocking=True)
        # for i, c in enumerate(class_mask[task_id]):
        #     for j, t in enumerate(target):
        #         if t == c:
        #             p = i * int(args.pool_size/args.ways)
        #             l = i * int(args.pool_size/args.ways) + int(args.pool_size/args.ways)
        #             prompts_matrix[j][p:l] = 1

        output = model(input, task_id=task_id, cls_features=cls_features, train=is_training,prompts_matrix=prompts_matrix)
        logits = output['logits']
        reps=output['pre_logits']
        for i, t in enumerate(target.tolist()):
            if t in e_reps_dict:
                e_reps_dict[t]=torch.cat((e_reps_dict[t],reps[i].unsqueeze(0)),dim=0)
            else:
                e_reps_dict[t] = reps[i].unsqueeze(0)
            if t in b_reps_dict:
                b_reps_dict[t]=torch.cat((b_reps_dict[t],reps[i].unsqueeze(0)),dim=0)
            else:
                b_reps_dict[t] = reps[i].unsqueeze(0)
                if epoch == 0:
                    R_dict[t] = args.R
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.n_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        if epoch==0:
            if args.pull_constraint and 'reduce_sim' in output:
                loss = loss - args.pull_constraint_coeff * output['reduce_sim']
        else:
            d_loss=dis_loss(b_reps_dict,args.Lambda,args.Alpha,args.Beta,args.M)/len(b_reps_dict)
            d_loss=d_loss.to(device, non_blocking=True)
            if args.pull_constraint and 'reduce_sim' in output:
                loss = loss - args.pull_constraint_coeff * output['reduce_sim']+d_loss

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # update anchor for each class
    for k,v in e_reps_dict.items():
        if k in anchor_dict:
            anchor_dict[k] = torch.mean(v,dim=0)
            # anchor_dict[k] = (anchor_dict[k]+torch.mean(v,dim=0))/2
        else:
            anchor_dict[k] = torch.mean(v,dim=0)
    # update radius for each class
    for k1, v1 in e_reps_dict.items():
        neg_m = torch.empty(0).to(device, non_blocking=True)
        for k2, v2 in e_reps_dict.items():
            if k1 != k2:
                d=torch.cdist(anchor_dict[k1].unsqueeze(0), v2).view(-1)
                neg_m= torch.cat((neg_m,(d - args.M)))
        R_dict[k1]=torch.quantile(neg_m, q=args.quan)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
            device, test_task_id=-1,task_id=-1, class_mask=None,args=None,):
    criterion = torch.nn.CrossEntropyLoss()
    header = 'Test: [Task {}]'.format(test_task_id)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = original_model(input)
            cls_features = output['pre_logits']

            output = model(input, task_id=task_id, cls_features=cls_features, train=False)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    metric_logger.synchronize_between_processes()
    test_result = '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
        top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss'])
    print(test_result)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((4, args.n_tasks))  # 3 for Acc@1, Acc@5, Loss
    for i in range(task_id + 1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, test_task_id=i,task_id=task_id, class_mask=class_mask, args=args)
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        acc_matrix[i, task_id] = test_stats['Acc@1']
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    diagonal = np.diag(acc_matrix)
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device,
                    class_mask=None, args = None,):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.n_tasks, args.n_tasks))
    for task_id in range(args.n_tasks):
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, device=device,
                            epoch=epoch, max_norm=args.clip_grad, is_training=True, task_id=task_id,
                            class_mask=class_mask, args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args)
        print(R_dict)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')