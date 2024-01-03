# 输入参数
GPU=${1:-'0'}
node=${2:-'1'}
bs=${3:-'8'}
dataset_file=${4:-'coco'}
epochs=${5:-'300'}
train_val=${6:-'train'}
resume=${7:-'/data/home/homefun/weights/Faster-RCNN/checkpoint.pth'}


if [ "$dataset_file" = 'coco' ]; then
    coco_path='/DATA/coco/images'
elif [ "$dataset_file" = 'voc' ]; then
    coco_path='/DATA/voc/VOCdevkit/VOC2012'
elif [ "$dataset_file" = 'voctest' ]; then
    coco_path='/DATA/voc/VOCdevkit/VOCtest'
fi

conda activate detr

cd ~/Faster-RCNN

time=$(date "+%Y-%m-%d-%H:%M:%S")

if [[ $GPU == *","* ]]; then
    if [ "$train_val" = 'train' ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
                                    --nproc_per_node=${node} \
                                    --use_env main.py \
                                    --resume ${resume} \
                                    --output_dir ~/weights/Faster-RCNN \
                                    --dataset_file ${dataset_file}\
                                    --coco_path ~${coco_path} \
                                    --data_path ~${coco_path} \
                                    --batch_size ${bs} \
                                    --epochs ${epochs} | tee ~/weights/Faster-RCNN/Log/${time}.txt 2>&1 &
    else 
        CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
                                    --nproc_per_node=${node} \
                                    --use_env main.py \
                                    --resume ${resume} \
                                    --output_dir ~/weights/Faster-RCNN/val \
                                    --dataset_file ${dataset_file}\
                                    --coco_path ~${coco_path} \
                                    --data_path ~${coco_path} \
                                    --batch_size ${bs} \
                                    --eval
    fi
else
    if [ "$train_val" = 'train' ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
                                    --resume ${resume} \
                                    --output_dir ~/weights/Faster-RCNN \
                                    --dataset_file ${dataset_file}\
                                    --coco_path ~${coco_path} \
                                    --data_path ~${coco_path} \
                                    --batch_size ${bs} \
                                    --epochs ${epochs} | tee ~/weights/Faster-RCNN/Log/${time}.txt 2>&1 &
    else 
        CUDA_VISIBLE_DEVICES=${GPU} python main.py \
                                    --resume ${resume} \
                                    --output_dir ~/weights/Faster-RCNN/val \
                                    --dataset_file ${dataset_file}\
                                    --coco_path ~${coco_path} \
                                    --data_path ~${coco_path} \
                                    --batch_size ${bs} \
                                    --eval
    fi
fi

# source ~/Faster-RCNN/startsh/start.sh 0 1 6 voc 30 train 
# source ~/Faster-RCNN/startsh/start.sh 0 1 6 voc 1 val 

# tensorboat
# 本地终端登陆远程服务器
# ssh -L 本地端口:127.0.0.1:TensorBoard端口 用户名@服务器的IP地址 -p 服务器登录端口
# 本地端口：查看 tensorboard 结果时，在浏览器中输入地址时的端口号
# TensorBoard端口：运行Tensorboard时指定的端口（默认为6006）
# 服务器登陆端口：登录服务器时指定的端口（默认为22）
# ssh -L 10086:127.0.0.1:8080 homefun@192.168.153.144

# 远程服务器中找到tensorboard所在目录并运行
# tensorboard --logdir ./tensorboard --port 8080

# 在本地浏览器中输入如下地址即可查看tensorboard结果
# -port 