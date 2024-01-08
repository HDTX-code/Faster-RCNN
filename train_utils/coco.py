# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from pycocotools import mask as coco_mask

from train_utils import trans as T
import transforms


class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms, cam_folder=None):
        self.img_folder = img_folder
        with open(ann_file, 'r') as file:
            self.data = json.load(file)
        self.ids = [j["id"] for j in self.data['images']]
        self.images = {j["id"] : j for j in self.data['images']}
        self.anns = self.get_anns(self.data["annotations"])
        self.cam_folder = cam_folder
        self._transforms = transforms
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = Image.open(os.path.join(self.img_folder, self.images[image_id]["file_name"])).convert("RGB")
        target = self.get_tensor(self.anns[image_id])
        # if self.cam_folder is not None:
        #     camdata = self.get_cam(image_id, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # if self.cam_folder is not None:
        #     input_tensor = torch.as_tensor(camdata, dtype=torch.float32)[None, :, :, :]
        #     downsampled_tensor = F.interpolate(input_tensor, size=(16, 16), mode='bilinear', align_corners=False)
        #     target['camdata'] = downsampled_tensor[0, :, :, :]
        return img, target
    
    def get_anns(self, anns):
        anndict = {}
        for ann in anns:
            if ann["image_id"] in anndict:
                anndict[ann["image_id"]].append(ann)
            else:
                anndict[ann["image_id"]] = [ann]
        return anndict
    
    # def get_cam(self, image_id, target):
    #     camdata = []
    #     for t in target:
    #         path = os.path.join(self.cam_folder, str(self.images[image_id]).split('.')[0] + "_" + str(t["category_id"] - 1) + '.npy')
    #         camdata.append(np.load(path)[np.newaxis, :, :])
    #     return np.concatenate(camdata, axis=0)
    
    def get_tensor(self, targets):
        boxes = []
        labels = []
        iscrowd = []
        area = []
        tar = {}
        for target in targets:
            boxes.append([float(x) for x in target["bbox"]])
            labels.append(target["category_id"])
            iscrowd.append(target["iscrowd"])
            area.append(target["area"])
        tar["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        tar["boxes"][:, 2:] += tar["boxes"][:, :2]
        tar["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        tar["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        tar["area"] = torch.as_tensor(area, dtype=torch.int64)
        tar["image_id"] = torch.tensor([target["image_id"]])
        return tar
    
    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        image_id = self.ids[idx]
        target = self.get_tensor(self.anns[image_id])
        data_height = int(self.images[image_id]["height"])
        data_width = int(self.images[image_id]["width"])

        return (data_height, data_width), target

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=data_transform(image_set))
    return dataset

def build_voc(image_set, args):
    datapath = Path(args.data_path)
    jsonpath = Path(args.coco_path)
    assert datapath.exists(), f'provided COCO path {datapath} does not exist'
    mode = 'JPEGImages'
    PATHS = {
        "train": (datapath / mode, jsonpath / 'train.json'),
        "val": (datapath / mode, jsonpath / 'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=data_transform(image_set), cam_folder=args.cam_path)
    return dataset
