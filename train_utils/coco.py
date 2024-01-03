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


class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cam_folder=None):
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
        


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_voc_transforms(image_set, size=800):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_voc_transforms(image_set, size=args.size), return_masks=args.masks)
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
    dataset = CocoDetection(img_folder, ann_file, transforms=make_voc_transforms(image_set, size=args.size), return_masks=False, cam_folder=args.cam_path)
    return dataset
