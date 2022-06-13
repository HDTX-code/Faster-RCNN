"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np

from backbone.creat_model import get_model
from utils import transforms
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset
from utils.utils import get_classes, get_dataset


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(args):
    # num_classes class_names max_map min_loss 初始化
    class_names, num_classes = get_classes(args.class_path)
    # num_workers
    num_workers = min(min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8]),
                      args.num_workers)  # number of workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    with open(args.val, encoding='utf-8') as f:
        val_lines = f.readlines()

    # 寻找类别最多的图
    # num_labels_max = 0
    # num_labels_max_id = 0
    # index = 0
    # for lines in val_lines:
    #     lines = lines.split()
    #     box_and_label = np.array([np.array(list(map(int, box.split(',')))) for box in lines[1:len(lines) - 1]])
    #     num_labels = len(np.unique(box_and_label[:, -1]))
    #     if num_labels > num_labels_max:
    #         num_labels_max_id = index
    #         num_labels_max = num_labels
    #     index += 1
    # print(num_labels_max)
    # print(num_labels_max_id)

    category_index = dict(zip(range(1, len(class_names) + 1), class_names))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    print('Using %g dataloader workers' % num_workers)

    # load validation data set
    val_dataset = get_dataset(val_lines, class_names=class_names, train=False)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers,
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    # model初始化
    model = get_model(args.backbone, num_classes + 1, model_path="", pretrained=False).to(device)

    # 载入你自己训练好的模型权重
    assert os.path.exists(args.weights_path), "not found {} file.".format(args.weights_path)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open(os.path.join(args.save_dir, "weights/loss_20220611221726/record_mAP.txt"), "w") as f:
        record_lines = ["mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--class_path', type=str, default=r"weights/voc_classes.txt", help='classes_path')
    parser.add_argument('--num_workers', type=int, default=24, help="num_workers")
    parser.add_argument("--backbone", type=str, default="resnet50_fpn")
    parser.add_argument('--val', default='weights/val.txt', help='dataset root')
    parser.add_argument('--weights_path', default='weights/loss_20220611210525/resnet50_fpn.pth', type=str,
                        help='training weights')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')
    parser.add_argument('--save_dir', type=str, default="weights/loss_20220611210525", help='save_dir')

    args = parser.parse_args()
    main(args)
