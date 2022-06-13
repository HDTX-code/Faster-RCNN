import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms

from backbone.creat_model import get_model
from utils.draw_box_utils import draw_objs
from utils.utils import get_classes


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    # num_classes class_names max_map min_loss 初始化
    class_names, num_classes = get_classes(args.class_path)
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # model初始化
    model = get_model(args.backbone, num_classes + 1, model_path="", pretrained=False).to(device)

    # load train weights
    assert os.path.exists(args.weights_path), "{} file dose not exist.".format(args.weights_path)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.to(device)

    category_index = dict(zip(range(1, len(class_names) + 1), class_names))

    # load image
    original_img = Image.open(args.pic_path)

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        # plot_img.save("test_result.jpg")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--class_path', type=str, default=r"weights/voc_classes.txt", help='classes_path')
    parser.add_argument("--backbone", type=str, default="resnet50_fpn")
    parser.add_argument('--weights_path', default='weights/loss_20220611210525/resnet50_fpn.pth', type=str,
                        help='training weights')
    parser.add_argument('--pic_path', default='data/VOCdevkit/VOC2007/JPEGImages/2009_003351.jpg', type=str,
                        help='pic_path')
    args = parser.parse_args()
    main(args)
