#!/bin/bash

LOG="experiments/logs/zf_faster_rcnn_e2e_voc_ped_face.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

./tools/train_net_voc_ped_face.py --gpu 1 --solver models/voc_ped/ZF/zf_faster_rcnn_e2e_voc_ped_face/solver.prototxt --weights data/imagenet_models/ZF.v2.caffemodel --iters 300000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --set TRAIN.SCALES '[500]' TRAIN.MAX_SIZE 1000 TRAIN.SNAPSHOT_ITERS 10000 

#./tools/train_net_voc_ped.py --gpu 1 --solver models/voc_ped/ZF/zf_faster_rcnn_e2e_voc_ped/solver.prototxt --weights output/zf_faster_rcnn_e2e_voc_ped/zf_faster_rcnn_e2e_voc_ped_iter_20000a.caffemodel --iters 300000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --set TRAIN.SCALES '[500]' TRAIN.MAX_SIZE 1000 TRAIN.SNAPSHOT_ITERS 10000 
