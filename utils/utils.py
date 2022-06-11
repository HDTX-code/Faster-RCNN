import math
from functools import partial

import numpy as np
import torch

from utils import transforms
from utils.dataset import FRCNNDataset

# ---------------------------------------------------#
#   展示训练的参数
# ---------------------------------------------------#
from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


# ---------------------------------------------------#
#   获取dataset
# ---------------------------------------------------#
def get_dataset(lines, class_names, train=True):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    if train:
        dataset = FRCNNDataset(lines, train=train, transforms=data_transform["train"], class_names=class_names)
    else:
        dataset = FRCNNDataset(lines, train=train, transforms=data_transform["val"], class_names=class_names)
    return dataset


# ---------------------------------------------------#
#   加载model
# ---------------------------------------------------#
def load_model(model, model_path):
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location='cpu')
    a = {}
    no_load = 0
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                a[k] = v
            else:
                no_load += 1
        except:
            print("模型加载出错")
            no_load = -1
            pass
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print("No_load: {}".format(no_load))
    print('Finished!')
    return model


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# ---------------------------------------------------#
#   lr 下降函数
# ---------------------------------------------------#
def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(
            math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
        )
    return lr


def step_lr(lr, decay_rate, step_size, iters):
    if step_size < 1:
        raise ValueError("step_size must above 1.")
    n = iters // step_size
    out_lr = lr * decay_rate ** n
    return out_lr


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ---------------------------------------------------#
#   lr 下降函数
# ---------------------------------------------------#
def get_lr_fun(optimizer_type, batch_size, Init_lr, Min_lr, Epoch, lr_decay_type):
    # 判断当前batch_size，自适应调整学习率
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    #   获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    return lr_scheduler_func, Init_lr_fit, Min_lr_fit


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_dataloader_with_aspect_ratio_group(train_dataset, aspect_ratio_group_factor, batch_size, num_workers):
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # 统计所有图像高宽比例在bins区间中的位置索引
    group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
    # 每个batch图片从同一高宽比例区间中取
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
    gen = torch.utils.data.DataLoader(train_dataset,
                                      batch_sampler=train_batch_sampler,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=train_dataset.collate_fn)
    return gen
