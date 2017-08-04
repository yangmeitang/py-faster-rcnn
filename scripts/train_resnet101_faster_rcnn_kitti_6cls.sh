#!/bin/bash

LOG="experiments/logs/resnet101_faster_rcnn_kitti_6cls.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

./tools/train_net_kitti.py --gpu 0 --solver models/kitti/resnet101_faster_rcnn_kitti_6cls/solver.prototxt --weights data/imagenet_models/ResNet-101-model.caffemodel --iters 300000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --set TRAIN.SCALES '[600]' TRAIN.MAX_SIZE 1000 TRAIN.SNAPSHOT_ITERS 10000 

