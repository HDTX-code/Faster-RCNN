import argparse
import os
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from backbone.creat_model import get_model
from utils.plot_curve import plot_loss_and_lr, plot_map
from utils.train_one_epoch import train_one_epoch, evaluate
from utils.utils import show_config, get_dataset, get_classes, set_optimizer_lr, get_lr_fun, \
    get_dataloader_with_aspect_ratio_group


def main(args):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                       训练相关准备                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    log_dir = os.path.join(args.sd, "loss_" + str(time_str))
    # 用来保存coco_info的文件
    results_file = os.path.join(log_dir,
                                "results{}.txt".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
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
    # num_classes class_names max_map min_loss 初始化
    class_names, num_classes = get_classes(args.cp)
    max_map = 0
    min_loss = 1e3
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 参数传递
    backbone = str(args.bb)
    model_path = str(args.mp)
    Init_Epoch = float(args.ie)
    Freeze_Epoch = int(args.fe)
    UnFreeze_Epoch = int(args.ufe)
    Freeze_Train = True if args.ufe != 0 else False,
    Init_lr = float(args.ilr)
    Min_lr = float(args.ilr) * 0.01
    optimizer_type_Freeze = str(args.opt_t_F)
    optimizer_type_UnFreeze = str(args.opt_t_UnF)
    momentum = float(args.m)
    lr_decay_type_Freeze = str(args.lr_d_t_F)
    lr_decay_type_UnFreeze = str(args.lr_d_t_UnF)
    pretrained = bool(args.pre)
    eval_flag = bool(args.ef)
    weight_decay = int(args.wd)
    print_freq = int(args.pf)
    Freeze_batch_size = int(args.fbs)
    UnFreeze_batch_size = int(args.ufbs)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                 dataset dataloader model                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    with open(args.train, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset, val_dataset = get_dataset(train_lines,
                                             class_names=class_names, train=True), get_dataset(val_lines,
                                                                                               class_names=class_names,
                                                                                               train=False)

    # 是否按图片相似高宽比采样图片组成batch, 使用的话能够减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor != -1:
        gen_Freeze = get_dataloader_with_aspect_ratio_group(train_dataset, aspect_ratio_group_factor,
                                                            Freeze_batch_size, num_workers)
        gen_UnFreeze = get_dataloader_with_aspect_ratio_group(train_dataset, aspect_ratio_group_factor,
                                                              UnFreeze_batch_size, num_workers)
    else:
        gen_Freeze = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=Freeze_batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=num_workers,
                                                 collate_fn=train_dataset.collate_fn)
        gen_UnFreeze = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=UnFreeze_batch_size,
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

    # model初始化
    model = get_model(backbone, num_classes + 1, model_path=model_path, pretrained=pretrained).to(device)

    # 打印训练参数
    show_config(backbone=backbone, num_classes=num_classes, model_path=model_path, Init_Epoch=Init_Epoch,
                Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, Freeze_batch_size=Freeze_batch_size,
                UnFreeze_batch_size=UnFreeze_batch_size, Cuda=Cuda, GPU=torch.cuda.current_device(),
                Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type_Freeze=optimizer_type_Freeze,
                optimizer_type_UnFreeze=optimizer_type_UnFreeze, lr_decay_type_UnFreeze=lr_decay_type_UnFreeze,
                lr_decay_type_Freeze=lr_decay_type_Freeze, save_dir=log_dir, num_workers=num_workers, momentum=momentum,
                num_train=num_train, num_val=num_val, amp=args.amp, pretrained=pretrained,
                eval_flag=eval_flag,print_freq=print_freq,)

    # 获取lr下降函数
    lr_scheduler_func_Freeze, Init_lr_fit_Freeze, Min_lr_fit_Freeze = get_lr_fun(optimizer_type_Freeze,
                                                                                 Freeze_batch_size,
                                                                                 Init_lr,
                                                                                 Min_lr,
                                                                                 Freeze_Epoch,
                                                                                 lr_decay_type_Freeze)
    lr_scheduler_func_UnFreeze, Init_lr_fit_UnFreeze, Min_lr_fit_UnFreeze = get_lr_fun(optimizer_type_UnFreeze,
                                                                                       UnFreeze_batch_size,
                                                                                       Init_lr,
                                                                                       Min_lr,
                                                                                       UnFreeze_Epoch,
                                                                                       lr_decay_type_UnFreeze)
    # 记录loss lr map
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
        mean_loss, lr = train_one_epoch(model, optimizer, gen_Freeze, device, epoch,
                                        print_freq=print_freq, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        if eval_flag:
            # evaluate on the test dataset
            coco_info = evaluate(model, gen_val, device=device)

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # pascal mAP

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    for param in model.backbone.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]

    #   根据optimizer_type选择优化器
    optimizer = {
        'adam': optim.Adam(params, Init_lr_fit_UnFreeze, betas=(momentum, 0.999), weight_decay=0),
        'sgd': optim.SGD(params, Init_lr_fit_UnFreeze, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type_UnFreeze]

    # 解冻训练
    for epoch in range(Freeze_Epoch + 1, UnFreeze_Epoch + Freeze_Epoch + 1):
        set_optimizer_lr(optimizer, lr_scheduler_func_UnFreeze, epoch - Freeze_Epoch + 1)
        mean_loss, lr = train_one_epoch(model, optimizer, gen_UnFreeze, device, epoch,
                                        print_freq=print_freq, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        if eval_flag:
            # evaluate on the test dataset
            coco_info = evaluate(model, gen_val, device=device)

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # pascal mAP

            if val_map[-1] > max_map:
                torch.save(model.state_dict(), os.path.join(log_dir, "{}.pth".format(backbone)))
                print("Save best map {:.3f} and loss {:.4f}".format(val_map[-1], train_loss[-1]))
                max_map = val_map[-1]
        else:
            if train_loss[-1] < min_loss:
                torch.save(model.state_dict(), os.path.join(log_dir, "{}.pth".format(backbone)))
                print("Save best loss {:.4f}".format(train_loss[-1]))
                min_loss = train_loss[-1]

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, log_dir)

    if eval_flag:
        # plot mAP curve
        if len(val_map) != 0:
            plot_map(val_map, log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameter setting')
    parser.add_argument('--bb', type=str, default='resnet50_fpn', help='backbone')
    parser.add_argument('--cp', type=str, default=r"weights/voc_classes.txt", help='classes_path')
    parser.add_argument('--sd', type=str, default="weights", help='save_dir')
    parser.add_argument('--mp', type=str, default="", help='model_path')
    parser.add_argument('--GPU', type=int, default=5, help='GPU_ID')
    parser.add_argument('--train', type=str, default=r"weights/train.txt", help="train_txt_path")
    parser.add_argument('--val', type=str, default=r"weights/val.txt", help="val_txt_path")
    parser.add_argument('--opt_t_F', type=str, default='adam', help="optimizer_type_Freeze")
    parser.add_argument('--opt_t_UnF', type=str, default='adam', help="optimizer_type_UnFreeze")
    parser.add_argument('--fbs', type=int, default=14, help="Freeze_batch_size")
    parser.add_argument('--ufbs', type=int, default=28, help="UnFreeze_batch_size")
    parser.add_argument('--argf', type=int, default=3, help="aspect_ratio_group_factor")
    parser.add_argument('--lr_d_t_F', type=str, default='cos', help="lr_decay_type_Freeze,'step' or 'cos'")
    parser.add_argument('--lr_d_t_UnF', type=str, default='cos', help="lr_decay_type_UnFreeze,'step' or 'cos'")
    parser.add_argument('--nw', type=int, default=24, help="num_workers")
    parser.add_argument('--ilr', type=float, default=1e-4, help="max lr")
    parser.add_argument('--m', type=float, default=0.9, help="momentum")
    parser.add_argument('--wd', type=float, default=0, help="weight_decay，adam is 0")
    parser.add_argument('--fe', type=int, default=18, help="Freeze_Epoch")
    parser.add_argument('--ufe', type=int, default=32, help="UnFreeze_Epoch")
    parser.add_argument('--ie', type=int, default=0, help="Init_Epoch")
    parser.add_argument('--pf', default=100, type=int, help="print_freq")
    parser.add_argument('--ef', default=False, action='store_true', help="Whether to calculate map during training")
    parser.add_argument('--pre', default=False, action='store_true', help="pretrained")
    parser.add_argument('--amp', default=False, action='store_true', help="amp or Not")
    args = parser.parse_args()

    main(args)
