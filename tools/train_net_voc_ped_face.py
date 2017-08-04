#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 Institute for Infocomm Research
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhou Lubing
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

from datasets.voc_ped import voc_ped
import caffe
import argparse
import pprint
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt
#from scipy.misc import imread, imresize
import cv2

def vis_result(im, boxes):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))

    for i in xrange(boxes.shape[0]):
        box = boxes[i, :]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        ax.add_patch(
            plt.Rectangle((x, y),
                          w,
                          h, fill=False,
                          edgecolor='red', linewidth=3.5)
            )

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format('class', 'class',
                                                  0.5),
                  fontsize=14)
    ax.imshow(im, aspect='equal')
    plt.draw()
    plt.show()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_ped', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    #print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    #    print 'cfgfile: ', args.cfg_file
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    
    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    #ihdb = voc_ped('voc_ped', '/home/mrbing/dataset/VOC')
    ihdb = voc_ped('merged_voc_ped_face', '/home/mrbing/dataset/VOC')
    roidb = ihdb.load_roidb()
    ihdb.append_flipped_images_voc_ped(roidb)
    print '{:d} roidb entries'.format(len(roidb))

    #n = 17000
    #print 'roidb samples:'
    #print 'roidb keys:'
    #print type(roidb)
    #for key in roidb[n].keys():
    #    print key
    #print 'image: ', roidb[n]['image']
    #print 'width: ', roidb[n]['width']
    #print 'height: ', roidb[n]['height']
    #print 'flipped: ', roidb[n]['flipped']
    #print 'boxes: ', roidb[n]['boxes'].shape
    #print roidb[n]['boxes']
    #print 'gt_overlaps: ', roidb[n]['gt_overlaps'].shape
    #print roidb[n]['gt_overlaps']
    #print 'gt_classes: ', roidb[n]['gt_classes'].shape
    #print roidb[n]['gt_classes']
    #print 'seg_areas: ', roidb[n]['seg_areas'].shape
    #print roidb[n]['seg_areas']
    #print 'max_classes: ', roidb[n]['max_classes'].shape
    #print roidb[n]['max_classes']
    #print 'max_overlaps: ', roidb[n]['max_overlaps'].shape
    #print roidb[n]['max_overlaps']
    
    #im = cv2.imread(roidb[n]['image'])
    #boxes = roidb[n]['boxes']
    ##print 'image: ', roidb[n]['image']
    ##print 'flipped: ', roidb[n]['flipped']
    ##print 'nums: ', boxes.shape
    ##print 'classes: ', roidb[n]['gt_classes']
    #if roidb[n]['flipped']:
    #    im = cv2.flip(im, 1)
    #vis_result(im, boxes)


    #output_dir = 'output/zf_faster_rcnn_e2e_voc_ped'
    #output_dir = 'output/zf_faster_rcnn_e2e_voc_ped_face'
    #output_dir = 'output/zf_faster_rcnn_e2e_voc_ped_face_4cls'
    output_dir = 'output/zf_faster_rcnn_e2e_voc_ped_face_3cls'

    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
