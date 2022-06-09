import math
from functools import partial

import numpy as np
import torch

from utils import transforms
from utils.dataset import FRCNNDataset


# ---------------------------------------------------#
#   展示训练的参数
# ---------------------------------------------------#
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
