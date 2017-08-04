#!/bin/bash

LOG="experiments/logs/resnet101_rfcn_voc_ped_3cls_ohem.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python ./tools/train_net_voc_ped.py --gpu 0 --solver models/pascal_voc/ResNet-101/rfcn_end2end/solver_ohem.prototxt --weights models/imagenet/ResNet-101-model.caffemodel --iters 110000 --cfg experiments/cfgs/rfcn_end2end_ohem.yml
