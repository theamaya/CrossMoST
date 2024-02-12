'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * References: timm and beit
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
 * https://github.com/microsoft/unilm/blob/master/beit
'''
import clip
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import os
from contextlib import suppress
import random
from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.dataset_3d import *

from pathlib import Path
from collections import OrderedDict

from ema import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

import utils
from utils.utils import NativeScalerWithGradNormCount as NativeScaler

import warnings

from utils.utils import get_dataset
import models.CrossMoST_models_scanobject as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn

from engine_self_training import train_one_epoch, evaluate

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='ULIP MUST training and evaluation', add_help=False)
    parser.add_argument('--config', default='')
    parser.add_argument('--checkpoint', default='')
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')

    parser.add_argument('--pretrain_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)

    parser.add_argument('--pretrain_dataset_name', default='scanobjectnn_withbg_img_pcl_h5', type=str)
    parser.add_argument('--validate_dataset_name', default='scanobjectnn_withbg_img_pcl_h5', type=str)

    parser.add_argument('--use_height', action='store_true',
                        help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    parser.add_argument('--nb_classes', default=0, type=int, help='number of the classification types')
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)

    # Augmentation parameters
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Model
    parser.add_argument('--model', default='ULIP_MUST_PointBERT', type=str)

    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--clip_model', default='ViT-B/16', help='pretrained clip model name')
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073))
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711))
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-device/per-gpu')  # 64
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--mask', action='store_true')
    parser.set_defaults(mask=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9998, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # some args
    # parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--dataset_root', type=str, default='/home/amaya/repos/CLIP2Point/data', help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--test', action='store_false')
    parser.set_defaults(test=False)
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')

    parser.add_argument('--text_prompt', type=str, default='This is a ', help='test freq')

    parser.add_argument('--VL', type=str, default='SLIP', help='vision-language model')
    parser.add_argument('--slip_model', type=str,
                        default='checkpoints/slip_base_100ep.pt',
                        help='vision-language model')
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--slip_model_name', default='SLIP_VITB16', type=str)

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.set_defaults(wandb=False)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--ulip', action='store_true')
    parser.set_defaults(ulip=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--pc_loss_weight', type=float, default=1, help='')
    parser.add_argument('--image_pc_align', action='store_true')
    parser.set_defaults(image_pc_align=True)

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

best_acc1 = 0
def main(args):
    utils.init_distributed_mode(args)
    config = cfg_from_yaml_file(args.config)

    global best_acc1

    if not args.output_dir:
        args.output_dir = os.path.join('output', args.dataset)
        if args.mask:
            args.output_dir = os.path.join(args.output_dir,
                                           "%s_mpatch%d_mratio%.1f_walign%.1f_tau%.1f_epoch%d_lr%.5f" % (
                                               args.clip_model[:5], config['mask_patch_size'],
                                               config['mask_ratio'], config['w_align'],
                                               config['conf_threshold'], config['epochs'], config['lr']))
        else:
            args.output_dir = os.path.join(args.output_dir, "%s_tau%.1f_epoch%d_lr%.5f" % (
                args.clip_model[:5], config['conf_threshold'], config['epochs'], config['lr']))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")

    tokenizer = SimpleTokenizer()

    dataset_train = get_dataset(None, tokenizer, args, 'train', config)
    dataset_val = get_dataset(None, tokenizer, args, 'test', config)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.output_dir)
    else:
        log_writer = None
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(dict(args._get_kwargs())) + "\n")

    args.batch_size = config['batch_size']
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(sampler_train is None),
        num_workers=args.workers, pin_memory=True, sampler=sampler_train, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=(sampler_val is None),
        num_workers=args.workers, pin_memory=True, sampler=sampler_val, drop_last=False)

    # create the model
    print("=> creating model: {}".format(args.model))
    classes = dataset_train.classes
    with open('../data/templates.json') as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    args.from_scratch = config.from_scratch
    args.image_pc_align = config.image_pc_align
    args.entropy_image = config.entropy_image
    args.entropy_pc = config.entropy_pc
    args.combined_pseudolabels = config.combined_pseudolabels


    model = models.CrossMoST_PointBERT(args, classes, templates, tokenizer)
    model.cuda(args.gpu)
    model.init_classifier(args)

    print("=> Setting learnable parameters for model: {}".format(args.model))

    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        resume='')
    print("Using EMA with decay = %.5f" % (args.model_ema_decay))

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train)

    args.lr = config['lr'] * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.epochs = config['epochs']
    args.eval_freq = config['eval_freq']
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))

    num_layers = 12  # model_without_ddp.model.visual.transformer.layers
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    for k, p in model.named_parameters():
        if 'point_encoder.encoder' in k:
            p.requires_grad=False
        if 'point_encoder.dvae' in k:
            p.requires_grad=False

    if config['only_image']:
        for k, p in model.named_parameters():
            if 'point_encoder' in k or 'pc_projection' in k:
                p.requires_grad=False
            else:
                p.requires_grad=True
    if config['only_pc']:
        for k, p in model.named_parameters():
            if 'visual' in k or 'image_projection' in k:
                p.requires_grad=False
            else:
                p.requires_grad=True

    for k, p in model.named_parameters():
        if p.requires_grad:
            print(k, ' is a learnable parameter')

    optimizer = create_optimizer(
        args, model_without_ddp,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    if args.amp:
        loss_scaler = NativeScaler()
        amp_autocast = torch.cuda.amp.autocast
    else:
        loss_scaler = None
        amp_autocast = suppress

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            temp = {}
            for key, value in checkpoint['model'].items():
                k = "module." + key
                temp[k] = value
            model.load_state_dict(temp, strict=True)

        test_stats = evaluate(data_loader_val, model, device, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1_image']:.1f}%")
        print(f"Accuracy of the network on the {len(dataset_val)} test pcs: {test_stats['acc1_pc']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    if utils.is_main_process() and args.wandb:
        wandb_id = args.run_id
        wandb.init(project='ULIP_MUST', id=wandb_id, config=args, reinit=True, save_code=True)



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        if config['mask']:
            args.mask= True
        else:
            args.mask= False


        train_stats = train_one_epoch(
            model, args, config,
            data_loader_train, optimizer, amp_autocast, device, epoch, loss_scaler,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            model_ema=model_ema,
        )

        if args.output_dir and utils.is_main_process() and (epoch + 1) % args.eval_freq == 0:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

            test_stats = evaluate(data_loader_val, model, device, model_ema=model_ema, args=args)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1_image']:.1f}%")
            print(f"Accuracy of the network on the {len(dataset_val)} test pcs: {test_stats['acc1_pc']:.1f}%")
            if max_accuracy < test_stats["acc1_pc"]:
                max_accuracy = test_stats["acc1_pc"]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1_pc'], head="test", step=epoch)
                log_writer.update(test_ema_acc1=test_stats['ema_acc1_pc'], head="test", step=epoch)
                log_writer.flush()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if utils.is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                    # wandb.watch(model)
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    test_stats = evaluate(data_loader_val, model, device, model_ema=model_ema, args=args)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1_image']:.1f}%")
    print(f"Accuracy of the network on the {len(dataset_val)} test pcs: {test_stats['acc1_pc']:.1f}%")
    if max_accuracy < test_stats["acc1_pc"]:
        max_accuracy = test_stats["acc1_pc"]
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

    print(f'Max accuracy: {max_accuracy:.2f}%')


if __name__ == '__main__':
    args = get_args()
    main(args)
