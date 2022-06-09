import torch
import torchvision
from torch.utils import model_zoo
from torchvision.models.feature_extraction import create_feature_extractor

from backbone.feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from network.rpn_function import AnchorsGenerator
from utils.utils import load_model


def creat_model_with_fpn(backbone, num_classes, model_path=None, pretrained=True):
    if backbone == 'eff_b0_fpn':
        # --- efficientnet_b0 fpn backbone --- #
        backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
        # print(backbone)
        return_layers = {"features.3": "0",  # stride 8
                         "features.4": "1",  # stride 16
                         "features.8": "2"}  # stride 32
        # 提供给fpn的每个特征层channel
        in_channels_list = [40, 80, 1280]
        new_backbone = create_feature_extractor(backbone, return_layers)

    elif backbone == 'mobil_v3_l_fpn':
        # --- mobilenet_v3_large fpn backbone --- #
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        # print(backbone)
        return_layers = {"features.6": "0",  # stride 8
                         "features.12": "1",  # stride 16
                         "features.16": "2"}  # stride 32
        # 提供给fpn的每个特征层channel
        in_channels_list = [40, 112, 960]
        new_backbone = create_feature_extractor(backbone, return_layers)

    else:
        # --- resnet50 fpn backbone --- #
        backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=3)
        model = FasterRCNN(backbone=backbone, num_classes=91)
        if pretrained:
            state_dict = model_zoo.load_url(
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if len(missing_keys) != 0 or len(unexpected_keys) != 0:
                print("missing_keys: ", len(missing_keys))
                print("unexpected_keys: ", len(unexpected_keys))
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        if model_path != "":
            model = load_model(model, model_path)
        return model
    # 检查新网络形状
    # img = torch.randn(1, 3, 224, 224)
    # outputs = new_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    if model_path is not None:
        model = load_model(model, model_path)

    return model


def creat_model_without_fpn(backbone, num_classes, model_path=None, pretrained=True):
    if backbone == 'vgg16':
        # vgg16
        backbone = torchvision.models.vgg16_bn(pretrained=pretrained)
        # print(backbone)
        backbone = create_feature_extractor(backbone, return_nodes={"features.42": "0"})
        # out = backbone(torch.rand(1, 3, 224, 224))
        # print(out["0"].shape)
        backbone.out_channels = 512

    elif backbone == 'eff_b0':
        # EfficientNetB0
        backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
        # print(backbone)
        backbone = create_feature_extractor(backbone, return_nodes={"features.5": "0"})
        # out = backbone(torch.rand(1, 3, 224, 224))
        # print(out["0"].shape)
        backbone.out_channels = 112

    else:
        # resnet50 backbone
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        # print(backbone)
        backbone = create_feature_extractor(backbone, return_nodes={"layer3": "0"})
        # out = backbone(torch.rand(1, 3, 224, 224))
        # print(out["0"].shape)
        backbone.out_channels = 1024

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    if model_path != '':
        model = load_model(model, model_path)

    return model


def get_model(backbone, num_classes, model_path=None, pretrained=True):
    if backbone.split("_")[-1] == 'fpn':
        model = creat_model_with_fpn(backbone, num_classes, model_path, pretrained)
    else:
        model = creat_model_without_fpn(backbone, num_classes, model_path, pretrained)
    return model.train()


if __name__ == '__main__':
    pass
