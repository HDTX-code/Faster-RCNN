import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from lxml import etree
from torch.utils.data.dataset import Dataset


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, class_names, train=True, transforms=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.train = train
        self.transforms = transforms
        self.class_dict = dict(zip(class_names, range(1, len(class_names) + 1)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        self.xml_list = []
        line = self.annotation_lines[index].split()
        image_path = line[0]
        box_and_label = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:len(line)-1]])
        image = Image.open(image_path)
        boxs, labels = box_and_label[:, :-1], box_and_label[:, -1]
        area = self.get_area(boxs)
        target = {
            "boxes": torch.as_tensor(boxs, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([index]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.zeros(area.shape, dtype=torch.int64)
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    @staticmethod
    def get_area(boxs):
        assert boxs.shape[-1] == 4 and len(boxs.shape) == 2
        h = boxs[:, 3] - boxs[:, 1]
        w = boxs[:, 2] - boxs[:, 0]
        return h * w

    def get_height_and_width(self, index):
        line = self.annotation_lines[index].split()
        hw = line[-1]
        data_height, data_width = [int(x) for x in hw.split(",")]
        return data_height, data_width

    def coco_index(self, index):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
            :param index: 索引
        """
        line = self.annotation_lines[index].split()
        box_and_label = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:len(line) - 1]])
        line = self.annotation_lines[index].split()
        hw = line[-1]
        data_height, data_width = [int(x) for x in hw.split(",")]
        boxs, labels = box_and_label[:, :-1], box_and_label[:, -1]
        area = self.get_area(boxs)
        target = {
            "boxes": torch.as_tensor(boxs, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([index]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.zeros(area.shape, dtype=torch.int64)
        }

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
