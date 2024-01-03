from .group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from .distributed_utils import init_distributed_mode, save_on_master, mkdir
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .coco import build_voc, build
from .trans import * 

import torch.utils.data
import torchvision

ap_ar_list = [  
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', 
                'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', 
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', 
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', 
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', 
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
            ]


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build(image_set, args)
    if args.dataset_file == 'voc' or args.dataset_file == 'voctest':
        return build_voc(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def save_ap_ar(data_list, writer, epoch):
    assert len(data_list) == len(ap_ar_list)
    for k, v in zip(ap_ar_list, data_list):
        writer.add_scalar(k, v, epoch)
        
def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)