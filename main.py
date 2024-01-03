import argparse
from pathlib import Path
import time
import os
import datetime

import random
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network_files.faster_rcnn_framework import FastRCNNPredictor, FasterRCNN

import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, build_dataset, collate_fn, create_aspect_ratio_groups, init_distributed_mode, save_ap_ar, save_on_master, mkdir
from train_utils.distributed_utils import get_rank

from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output_dir', default="/data/home/homefun/weights/Faster-RCNN",
                        help='path where to save, empty for no saving')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[7, 12], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # 训练的batch size
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='voc')
    parser.add_argument('--coco_path', default="/data/home/homefun/DATA/voc/VOCdevkit/VOCtest", type=str)
    parser.add_argument('--data_path', default="/data/home/homefun/DATA/voc/VOCdevkit/VOCtest", type=str)
    parser.add_argument('--cam_path', default="/data/home/homefun/weights/CAM/campic/gradcam", type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--size', default=800, type=int)
    parser.add_argument('--cam_channel', default=2048, type=int)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default="", help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    return parser

def create_model(args):
    # 如果显存很小，建议使用默认的FrozenBatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load(args.pretrain_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes + 1)

    return model

def main(args):
    init_distributed_mode(args)
    device = torch.device(args.device)
    
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load train data set
    print("Creating data loaders")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # if args.aspect_ratio_group_factor >= 0:
    #     # 统计所有图像比例在bins区间中的位置索引
    #     group_ids = create_aspect_ratio_groups(dataset_train, k=args.aspect_ratio_group_factor)
    #     train_batch_sampler = GroupedBatchSampler(sampler_train, group_ids, args.batch_size)
    # else:
    train_batch_sampler = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler,
                                collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    print("Creating model")
    model = create_model(args)
    model.to(device)
   
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if os.path.exists(args.resume):
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if args.eval:
        utils.evaluate(model, data_loader_val, device=device)
        return

    best_map = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader_train,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)
        writer.add_scalar("mean_loss", mean_loss.item(), epoch)
        writer.add_scalar("lr", lr, epoch)

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        coco_info = utils.evaluate(model, data_loader_val, device=device)
        save_ap_ar(coco_info, writer, epoch) # pascal mAP

        if args.output_dir:
            if coco_info[1] > best_map or epoch == args.epochs - 1:
                # 只在主节点上执行保存权重操作
                save_files = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch}
                if args.amp:
                    save_files["scaler"] = scaler.state_dict()
                save_on_master(save_files,
                            os.path.join(args.output_dir, 'model_{}.pth'.format(coco_info[1])))
                best_map = coco_info[1]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Faster training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # output_dir
    now = datetime.datetime.now()
    time_now = now.strftime(r"%Y%m%d%H%M")
    args.pretrain_path = os.path.join(args.output_dir, "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
    args.output_dir = os.path.join(args.output_dir, time_now)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir)
    main(args)
    writer.close()