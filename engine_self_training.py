import math
import sys
from typing import Iterable
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import kl_divergence
import utils
from utils.utils import *
from timm.utils import accuracy

def train_one_epoch(model: torch.nn.Module, args, train_config,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    for step, ((images_weak, images_strong, mask, pc_weak, pc_strong, pc_mask), targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay
        model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)

        images_weak, images_strong = images_weak.to(device, non_blocking=True), images_strong.to(device, non_blocking=True)
        pc_weak, pc_strong = pc_weak.to(device, non_blocking=True), pc_strong.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        try:
            combine_naive= train_config['naive']
        except:
            combine_naive= False

        with torch.no_grad():
            # pseudo-label with ema model
            image_logits, pc_logits = model_ema.ema(images_weak, pc_weak)
            probs_ema_image = F.softmax(image_logits, dim=-1)
            probs_ema_pc = F.softmax(pc_logits, dim=-1)

            score_image, pseudo_targets_image = probs_ema_image.max(-1)
            score_pc, pseudo_targets_pc = probs_ema_pc.max(-1)

            b = (1 / probs_ema_image.shape[1]) * torch.ones(probs_ema_image.shape).cuda()
            loss_entropy_image = -kl_divergence(probs_ema_image, b)
            loss_entropy_pc = -kl_divergence(probs_ema_pc, b)

            if train_config['combined_pseudolabels']:
                score_pc = score_pc* train_config['conf_weight_pc']

                if train_config['agreement_pseudolabels']:
                    combined_scores = torch.min(score_pc, score_image)
                    conf_mask = (pseudo_targets_pc == pseudo_targets_image)*(combined_scores > train_config['agreement_pseudolabels_min_thresh'])
                    conf_mask_image = conf_mask
                    conf_mask_pc = conf_mask_image
                    pseudolabel_agreement_loss = (pseudo_targets_image[conf_mask_image] != pseudo_targets_pc[
                        conf_mask_pc]).sum() / pseudo_targets_image[conf_mask_image].shape[0]


                else:
                    if combine_naive:
                        bs = score_pc.shape[0]
                        picked = torch.randint(2, (bs,)).cuda()
                        combined_scores = score_pc * (picked == 1) + score_image * (picked == 0)
                    else:
                        combined_scores = torch.max(score_pc, score_image)
                    combined_targets = pseudo_targets_pc * (combined_scores == score_pc) + pseudo_targets_image * (combined_scores == score_image)

                    conf_mask_image = combined_scores > train_config['conf_threshold_combined']
                    conf_mask_pc = conf_mask_image

                    pseudolabel_agreement_loss = (pseudo_targets_image[conf_mask_image]!=pseudo_targets_pc[conf_mask_pc]).sum()/pseudo_targets_image[conf_mask_image].shape[0]
                    pseudo_targets_image = combined_targets
                    pseudo_targets_pc = combined_targets

            else:
                conf_mask_image = score_image > train_config['conf_threshold_image']
                conf_mask_pc = score_pc > train_config['conf_threshold_pc']

            pseudo_label_acc_image = (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image]).float().mean().item()
            conf_ratio_image = conf_mask_image.float().sum()/conf_mask_image.size(0)
            if train_config['from_scratch']:
                pseudo_label_acc_pc = (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image]).float().mean().item()
            else:
                pseudo_label_acc_pc = (pseudo_targets_pc[conf_mask_pc] == targets[conf_mask_pc]).float().mean().item()
            conf_ratio_pc = conf_mask_pc.float().sum() / conf_mask_pc.size(0)

            metric_logger.update(conf_ratio_image=conf_ratio_image)
            metric_logger.update(pseudo_label_acc_image=pseudo_label_acc_image)
            metric_logger.update(conf_ratio_pc=conf_ratio_pc)
            metric_logger.update(pseudo_label_acc_pc=pseudo_label_acc_pc)


        with amp_autocast():
            if args.mask:
                logits_image, logits_pc, loss_mim_image, loss_mim_pc, loss_align_image, loss_align_pc, pc_image_align_logits, image_pc_align_logits = model(images_strong, pc_strong, Mask=mask)
            else:
                logits_image, logits_pc = model(images_strong, pc_strong)

            # self-training loss
            if train_config['trans_pcl_img']:
                loss_st_image = F.cross_entropy(logits_image[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
            else:
                loss_st_image = F.cross_entropy(logits_image[conf_mask_image], pseudo_targets_image[conf_mask_image])

            if train_config['from_scratch']:
                loss_st_pc = F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
            elif train_config['trans_img_pcl']:
                loss_st_pc = F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
            else:
                loss_st_pc = F.cross_entropy(logits_pc[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])

            # fairness regularization
            probs = F.softmax(logits_image,dim=-1)
            probs_all = all_gather_with_grad(probs)
            probs_batch_avg_image = probs_all.mean(0) # average prediction probability across all gpus

            probs = F.softmax(logits_pc, dim=-1)
            probs_all = all_gather_with_grad(probs)
            probs_batch_avg_pc = probs_all.mean(0)  # average prediction probability across all gpus

            probs_avg = probs_batch_avg_image
            loss_fair_image = -(torch.log(probs_avg)).mean()
            probs_avg = probs_batch_avg_pc
            loss_fair_pc = -(torch.log(probs_avg)).mean()

            if args.mask:
                labels = torch.eye(pc_image_align_logits.shape[0]).cuda()
                loss_pc_image_align = F.cross_entropy(pc_image_align_logits, labels)
                loss_image_pc_align = F.cross_entropy(image_pc_align_logits, labels)
                # global-local feature alignment loss
                loss_align_image = torch.mean(loss_align_image)
                loss_align_pc = torch.mean(loss_align_pc)

                if train_config['only_image']:
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image + train_config['w_mim_image']*loss_mim_image + train_config['w_align_image'] * loss_align_image
                elif train_config['only_pc']:
                    loss= loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_mim_pc']*loss_mim_pc + train_config['w_align_pc'] * loss_align_pc
                else:
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image + train_config['w_mim_image']*loss_mim_image + train_config['w_align_image'] * loss_align_image + args.pc_loss_weight * (loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_mim_pc']*loss_mim_pc +train_config['w_align_pc'] * loss_align_pc)

                    if train_config['image_pc_align']:
                        loss = loss + train_config['w_image_pc_align']*loss_pc_image_align + train_config['w_image_pc_align']*loss_image_pc_align
                    if train_config['pseudolabel_agreement_loss']:
                        loss = loss + train_config['w_pseudo_agree'] * pseudolabel_agreement_loss
                    if train_config['entropy_image']:
                        loss = loss + loss_entropy_image
                    if train_config['entropy_pc']:
                        loss = loss + loss_entropy_pc

            else:
                if train_config['only_image']:
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image
                elif train_config['only_pc']:
                    loss = loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc
                else:
                    loss = loss_st_image + loss_st_pc + train_config['w_fair_image'] * loss_fair_image + train_config['w_fair_pc'] * loss_fair_pc

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:
            loss.backward(create_graph=False)
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()

        metric_logger.update(loss_st_image=loss_st_image.item())
        metric_logger.update(loss_fair_image=loss_fair_image.item())
        metric_logger.update(loss_st_pc=loss_st_pc.item())
        metric_logger.update(loss_fair_pc=loss_fair_pc.item())
        metric_logger.update(loss_entropy_image=loss_entropy_image.item())
        metric_logger.update(loss_entropy_pc=loss_entropy_pc.item())
        if train_config['combined_pseudolabels']:
            metric_logger.update(loss_pseudolabel_agreement=pseudolabel_agreement_loss.item())

        if args.mask:
            metric_logger.update(loss_pc_image_align=loss_pc_image_align.item())
            metric_logger.update(loss_image_pc_align=loss_image_pc_align.item())
            metric_logger.update(loss_mim_image=loss_mim_image.item())
            metric_logger.update(loss_align_image=loss_align_image.item())
            metric_logger.update(loss_mim_pc=loss_mim_pc.item())
            metric_logger.update(loss_align_pc=loss_align_pc.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:
            log_writer.update(loss_st_image=loss_st_image.item(), head="train")
            log_writer.update(loss_fair_image=loss_fair_image.item(), head="train")
            log_writer.update(loss_st_pc=loss_st_pc.item(), head="train")
            log_writer.update(loss_fair_pc=loss_fair_pc.item(), head="train")

            if args.mask:
                log_writer.update(loss_mim_image=loss_mim_image.item(), head="train")
                log_writer.update(loss_align_image=loss_align_image.item(), head="train")
                log_writer.update(loss_mim_pc=loss_mim_pc.item(), head="train")
                log_writer.update(loss_align_pc=loss_align_pc.item(), head="train")

            log_writer.update(conf_ratio_image=conf_ratio_image, head="train")
            log_writer.update(pseudo_label_acc_image=pseudo_label_acc_image, head="train")
            log_writer.update(conf_ratio_pc=conf_ratio_pc, head="train")
            log_writer.update(pseudo_label_acc_pc=pseudo_label_acc_pc, head="train")

            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        loss_dict={}
        for k, v in metric_logger.meters.items():
            loss_dict[k] = metric_logger.meters.get(k).value
        if utils.utils.is_main_process() and args.wandb:
            wandb.log(loss_dict)

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None, args=None):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if model_ema is not None:
        model_ema.ema.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0][0].to(device, non_blocking=True)
        pcs = batch[0][1].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # images = batch[0].to(device, non_blocking=True)
        # pcs = batch[1].to(device, non_blocking=True)
        # target = batch[-1].to(device, non_blocking=True)

        # compute output
        output = model(images, pcs)

        acc_image = accuracy(output[0], target)[0]
        acc_pc = accuracy(output[1], target)[0]
        metric_logger.meters['acc1_image'].update(acc_image.item(), n=images.shape[0])
        metric_logger.meters['acc1_pc'].update(acc_pc.item(), n=images.shape[0])

        if model_ema is not None:
            ema_output = model_ema.ema(images, pcs)

            ema_acc1_image = accuracy(ema_output[0], target)[0]
            ema_acc1_pc = accuracy(ema_output[1], target)[0]
            metric_logger.meters['ema_acc1_image'].update(ema_acc1_image.item(), n=images.shape[0])
            metric_logger.meters['ema_acc1_pc'].update(ema_acc1_pc.item(), n=images.shape[0])

    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_image))
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_pc))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

