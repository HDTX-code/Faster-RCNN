import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, train=True, transforms=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        line = self.annotation_lines[index].split()
        image_path = line[0]
        box_and_label = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        image_id = line[0].split("/")[-1][:-4]
        image = Image.open(image_path)
        if self.train:
            boxs, labels = box_and_label[:, :-1], box_and_label[:, -1]
            area = self.get_area(boxs)
            target = {
                "boxes": boxs,
                "labels": labels,
                "image_id": image_id,
                "area": area,
            }
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target
        else:
            return image

    @staticmethod
    def get_area(boxs):
        assert boxs.shape[-1] == 4 and len(boxs.shape) == 2
        h = boxs[:, 3] - boxs[:, 1]
        w = boxs[:, 2] - boxs[:, 0]
        return h * w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
