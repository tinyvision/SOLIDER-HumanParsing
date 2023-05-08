#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import timeit
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.distributed  as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data

import networks
import utils.schp as schp
from datasets.datasets import LIPDataSet
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='/home/xianzhe.xxz/datasets/HumanParsing/LIP')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0,1,2')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    parser.add_argument("--syncbn", action="store_true", help='use syncbn or not')
    parser.add_argument("--imagenet", action="store_true", help='use syncbn or not')
    parser.add_argument("--optimizer", type=str, default='sgd', help='which optimizer to use')

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr_divider", type=int, default=100)
    parser.add_argument("--cyclelr_divider", type=int, default=2)

    return parser.parse_args()


def main():
    args = get_arguments()
    local_rank = args.local_rank

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        if local_rank == 0:
            os.makedirs(args.log_dir)
    if local_rank == 0:
        with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
            json.dump(vars(args), opt_file)
        print(args)
    #gpus = [int(i) for i in args.gpu.split(',')]
    #if not args.gpu == 'None':
    #    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dist.init_process_group(backend='nccl')

    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model Initialization
    if args.imagenet:
        convert_weights = True
    else:
        convert_weights = False
    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain, convert_weights=convert_weights)
    for name, param in model.named_parameters():
        #if name.startswith("backbone.patch_embed"):
        if "patch_embed" in name:
            print(name)
            param.requires_grad = False

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space

    restore_from = args.model_restore
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        checkpoint = torch.load(restore_from, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    model.to(device)
    if args.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    schp_model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain, convert_weights=convert_weights)
    #for name, param in schp_model.named_parameters():
        #if name.startswith("backbone.patch_embed"):
     #   if "patch_embed" in name:
      #      param.requires_grad = False

    if os.path.exists(args.schp_restore):
        print('Resuming schp checkpoint from {}'.format(args.schp_restore))
        schp_checkpoint = torch.load(args.schp_restore, map_location='cpu')
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = schp_checkpoint['cycle_n']
        schp_model.load_state_dict(schp_model_state_dict)

    schp_model.to(device)
    if args.syncbn:
        print('----use syncBN in model!----')
        schp_model = nn.SyncBatchNorm.convert_sync_batchnorm(schp_model)
    schp_model = DDP(schp_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)
    #criterion = DataParallelCriterion(criterion)
    #criterion.to(device)

    # Data Loader
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    train_dataset = LIPDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform)
    dist_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=dist_sampler,
                                   num_workers=8, pin_memory=False, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))

    # Optimizer Initialization
    if args.optimizer == 'sgd':
        print("using SGD optimizer")
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        print("using Adam optimizer")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Original warmup_epoch=10, changed to 3 for fix backbone finetune
    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                 eta_min=args.learning_rate / args.lr_divider, warmup_epoch=args.warmup_epochs,
                                 start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / args.cyclelr_divider,
                                 cyclical_epoch=args.cycle_epochs)

    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()
    iter_start = timeit.default_timer()

    model.train()
    for epoch in range(start_epoch, args.epochs):
        dist_sampler.set_epoch(epoch)
        lr = lr_scheduler.get_lr()[0]

        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader) * epoch
            images, labels, _ = batch
            #labels = labels.cuda(non_blocking=True)
            labels = labels.to(device)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)
#            for name, param in model.named_parameters():
#                print(name,': ', param.requires_grad)

            #print('fixed', model.state_dict()['module.conv1.weight'][0,0,0,0])
            #print('update', model.state_dict()['module.decoder.conv4.weight'][0,0,0,0])


            preds = model(images)

            # Online Self Correction Cycle with Label Refinement
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds = schp_model(images)
                    soft_fused_preds = soft_preds[0][-1]
                    soft_edges = soft_preds[1][-1]
                    soft_preds = soft_fused_preds
            else:
                soft_preds = None
                soft_edges = None

            loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_rank == 0 and i_iter % 100 == 0:
                print('iter = {} of {} completed, lr = {}, loss = {}, time = {}'.format(i_iter, total_iters, lr,
                                                                             loss.data.cpu().numpy(), (timeit.default_timer()-iter_start)/100))
                iter_start = timeit.default_timer()
        lr_scheduler.step()
        if local_rank == 0 and (epoch + 1) % (args.eval_epochs) == 0:
            schp.save_schp_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, False, args.log_dir, filename='checkpoint_{}.pth.tar'.format(epoch + 1))

        # Self Correction Cycle with Model Aggregation
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print('Self-correction cycle number {}'.format(cycle_n))
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)
            if local_rank == 0:
                schp.save_schp_checkpoint({
                    'state_dict': schp_model.state_dict(),
                    'cycle_n': cycle_n,
                }, False, args.log_dir, filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                             (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
