import argparse
import os
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from train_utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from utils.creat_model import get_model
from utils.plot_curve import plot_loss_and_lr, plot_map
from utils.train_one_epoch import train_one_epoch
from utils.utils import show_config, get_dataset, get_classes, get_lr_scheduler, set_optimizer_lr, get_lr_fun


def main(args):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                       训练相关准备                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
    # 检查保存文件夹是否存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                    训练参数设置相关准备                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    Cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_workers
    num_workers = min(min([os.cpu_count(), args.bs if args.bs > 1 else 0, 8]), args.nw)  # number of workers
    # 按图片相似高宽比采样区间数 采样器
    aspect_ratio_group_factor = args.argf
    train_batch_sampler = None
    # num_classes class_names
    class_names, num_classes = get_classes(args.cp)
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 参数传递
    backbone = str(args.bb)
    model_path = str(args.mp)
    Init_Epoch = float(args.ie)
    Freeze_Epoch = int(args.fe)
    UnFreeze_Epoch = int(args.ufe)
    batch_size = int(args.bs)
    Freeze_Train = True if args.ufe != 0 else False,
    Init_lr = float(args.ilr)
    Min_lr = float(args.ilr) * 0.01
    optimizer_type_Freeze = str(args.opt_t_F)
    optimizer_type_UnFreeze = str(args.opt_t_UnF)
    momentum = float(args.momentum)
    lr_decay_type_Freeze = str(args.lr_d_t_F)
    lr_decay_type_UnFreeze = str(args.lr_d_t_UnF)
    pretrained = bool(args.pre)
    eval_flag = bool(args.el)
    eval_period = int(args.ep)
    weight_decay = int(args.wd)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                 dataset dataloader model                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    with open(args.train, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset, val_dataset = get_dataset(train_lines, train=True), get_dataset(val_lines, train=False)
    # 是否按图片相似高宽比采样图片组成batch, 使用的话能够减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    if train_batch_sampler is not None:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        gen = torch.utils.data.DataLoader(train_dataset,
                                          batch_sampler=train_batch_sampler,
                                          pin_memory=True,
                                          num_workers=num_workers,
                                          collate_fn=train_dataset.collate_fn)
    else:
        gen = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=num_workers,
                                          collate_fn=train_dataset.collate_fn)

    gen_val = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=num_workers,
                                          collate_fn=val_dataset.collate_fn)

    model = get_model(backbone, num_classes, model_path=None, pretrained=True).to(device)

    # 打印训练参数
    show_config(backbone=backbone, num_classes=num_classes, model_path=model_path, Init_Epoch=Init_Epoch,
                Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, batch_size=UnFreeze_Epoch,
                Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type_Freeze=optimizer_type_Freeze,
                optimizer_type_UnFreeze=optimizer_type_UnFreeze, lr_decay_type_UnFreeze=lr_decay_type_UnFreeze,
                lr_decay_type_Freeze=lr_decay_type_Freeze, save_dir=log_dir, num_workers=num_workers, momentum=momentum,
                num_train=num_train, num_val=num_val, amp=args.amp, pretrained=pretrained, eval_flag=eval_flag,
                eval_period=eval_period, Cuda=Cuda, GPU=torch.cuda.current_device())

    lr_scheduler_func_Freeze, Init_lr_fit_Freeze, Min_lr_fit_Freeze = get_lr_fun(optimizer_type_Freeze,
                                                                                 batch_size,
                                                                                 Init_lr,
                                                                                 Min_lr,
                                                                                 Freeze_Epoch,
                                                                                 lr_decay_type_Freeze)
    lr_scheduler_func_UnFreeze, Init_lr_fit_UnFreeze, Min_lr_fit_UnFreeze = get_lr_fun(optimizer_type_UnFreeze,
                                                                                       batch_size,
                                                                                       Init_lr,
                                                                                       Min_lr,
                                                                                       UnFreeze_Epoch,
                                                                                       lr_decay_type_UnFreeze)

    train_loss = []
    learning_rate = []
    val_map = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone train 5 epochs                       #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 冻结训练
    for param in model.backbone.parameters():
        param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]

    #   根据optimizer_type选择优化器
    optimizer = {
        'adam': optim.Adam(params, Init_lr_fit_Freeze, betas=(momentum, 0.999), weight_decay=0),
        'sgd': optim.SGD(params, Init_lr_fit_Freeze, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type_Freeze]

    for epoch in range(1, Freeze_Epoch + 1):
        set_optimizer_lr(optimizer, lr_scheduler_func_Freeze, epoch - 1)
        mean_loss, lr = train_one_epoch(model, optimizer, gen,
                                        device, epoch, print_freq=50,
                                        warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    for param in model.backbone.parameters():
        param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]

    #   根据optimizer_type选择优化器
    optimizer = {
        'adam': optim.Adam(params, Init_lr_fit_UnFreeze, betas=(momentum, 0.999), weight_decay=0),
        'sgd': optim.SGD(params, Init_lr_fit_UnFreeze, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type_UnFreeze]

    # 解冻训练
    for epoch in range(Freeze_Epoch + 1, UnFreeze_Epoch + Freeze_Epoch + 1):
        set_optimizer_lr(optimizer, lr_scheduler_func_UnFreeze, epoch - Freeze_Epoch + 1)
        mean_loss, lr = train_one_epoch(model, optimizer, gen,
                                        device, epoch, print_freq=50,
                                        warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--bb', type=str, default='resnet50', help='backbone')
    parser.add_argument('--cp', type=str, default="./model_data/voc_classes.txt", help='classes_path')
    parser.add_argument('--save_dir', type=str, default="./weights", help='save_dir')
    parser.add_argument('--mp', type=str, default="weights/pre_train/resnet50-19c8e357.pth", help='model_path')
    parser.add_argument('--GPU', type=int, default=5, help='GPU_ID')
    parser.add_argument('--train', type=str, default="./2007_train.txt", help="train_txt_path")
    parser.add_argument('--val', type=str, default="./2007_val.txt", help="val_txt_path")
    parser.add_argument('--args.opt_t_F', type=str, default='adam', help="optimizer_type_Freeze")
    parser.add_argument('--args.opt_t_UnF', type=str, default='adam', help="optimizer_type_UnFreeze")
    parser.add_argument('--bs', type=int, default=36, help="batch_size")
    parser.add_argument('--lr_decay_type', type=str, default='cos', help="lr_decay_type,'step' or 'cos'")
    parser.add_argument('--nw', type=int, default=24, help="num_workers")
    parser.add_argument('--ilr', type=float, default=1e-4, help="max lr")
    parser.add_argument('--momentum', type=float, default=0.9, help="优化器动量")
    parser.add_argument('--wd', type=float, default=0, help="weight_decay，adam is 0")
    parser.add_argument('--Freeze_Epoch', type=int, default=30, help="Freeze_Epoch")
    parser.add_argument('--UnFreeze_Epoch', type=int, default=60, help="UnFreeze_Epoch")
    parser.add_argument('--Init_Epoch', type=int, default=0, help="Init_Epoch")
    parser.add_argument('--eval_period', type=int, default=5, help="eval_period")
    parser.add_argument('--eval_flag', default=False, action='store_true', help="是否在训练过程中检测")
    parser.add_argument('--pretrained', default=False, action='store_true', help="pretrained")
    parser.add_argument('--amp', default=False, action='store_true', help="amp")
    args = parser.parse_args()

    main(args)
