## 文件结构：
```
  ├── backbone: Faster-RCNN模型backbone部分，可选择res50、res50+fpn、Mobilenetv2、vgg
  ├── network：Faster-RCNN模型中的RPN、ROI head、RCNN head 部分
  ├── utils: Dataset train_one_epoch evaluate transformer等相关程序
  ├── weights: 保存训练权重，结果等文件
  ├── train.py: 开始训练的程序
  ├── predict.py：预测单张图片，并且输出推理单张图片的时间
  ├── validation.py：获得各个类别的map等参数
  ├── voc_xml_transform_txt.py：把voc数据集的xml文件转成训练所需要的txt文件并保存到指定位置
```
## 训练
* 确保提前准备好数据集
* 使用`voc_xml_transform_txt.py`转成训练所需的txt文件格式（例如:[train.txt](weights/train.txt), [val.txt](weights/val.txt)）
* [训练集+验证集统计情况](weights/data.out)
### Faster R-CNN
* 训练过程中的lr与loss：12轮次冻结backbone训练+24轮次解冻训练![lr and loss](weights/loss_20220611221726/loss_and_lr.png)
* 训练过程中的map(IoU=0.5)：![map(IoU=0.5)](weights/loss_20220611221726/mAP.png)
* [训练过程所有输出](weights/resnet50_3.out)
* [各个轮次的指标结果](weights/loss_20220611221726/results20220611221726.txt)
### Faster R-CNN + FPN
* 训练过程中的lr与loss：12轮次冻结backbone训练+24轮次解冻训练![lr and loss](weights/loss_20220611210525/loss_and_lr.png)
* 训练过程中的map(IoU=0.5)：![map(IoU=0.5)](weights/loss_20220611210525/mAP.png)
* [训练过程所有输出](weights/resnet50_fpn_3.out)
* [各个轮次的指标结果](weights/loss_20220611210525/results20220611210525.txt)
## 预测
* 需要修改`--class_path`指向记录类别的txt文件（例如[voc_classes.txt](weights/voc_classes.txt)）
* 需要修改`--pic_path`指向需要预测的图片地址（例如[predict.jpg](2009_003351.jpg)）
* 需要修改`--weights_path`指向训练保存的权重文件（例如[Faster R-CNN.pth](weights/loss_20220611221726/resnet50.pth),[Faster R-CNN + FPN.pth](weights/loss_20220611210525/resnet50_fpn.pth)）
* 需要修改`--backbone`与前面的权值文件对应
### Faster R-CNN
* `python predict.py --backbone resnet50 --weights_path weights/loss_20220611221726/resnet50.pth`
* 预测结果：![predict for Faster R-CNN](test_result_resnet50.jpg)
### Faster R-CNN + FPN
* `python predict.py --backbone resnet50_fpn --weights_path weights/loss_20220611210525/resnet50_fpn.pth`
* 预测结果：
![predict for Faster R-CNN + FPN](test_result_resnet50_fpn.jpg)
## 获得各个类别的map等参数
* 需要修改`--class_path`指向记录类别的txt文件（例如[voc_classes.txt](weights/voc_classes.txt)）
* 需要修改`--val`指向验证集/测试集txt文件（例如[val.txt](weights/val.txt)）
* 需要修改`--weights_path`指向训练保存的权重文件（例如[Faster R-CNN.pth](weights/loss_20220611221726/resnet50.pth),[Faster R-CNN + FPN.pth](weights/loss_20220611210525/resnet50_fpn.pth)）
* 需要修改`--backbone`与前面的权值文件对应
### Faster R-CNN
* `python validation.py --backbone resnet50 --weights_path weights/loss_20220611221726/resnet50.pth`
* 预测结果：[predict for Faster R-CNN](weights/loss_20220611221726/record_mAP.txt)
### Faster R-CNN + FPN
* `python validation.py --backbone resnet50_fpn --weights_path weights/loss_20220611210525/resnet50_fpn.pth`
* 预测结果：[predict for Faster R-CNN + FPN](weights/loss_20220611210525/record_mAP.txt)

