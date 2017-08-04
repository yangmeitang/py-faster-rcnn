# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from fast_rcnn.config import cfg

import PIL

class voc_ped(object):
    def __init__(self, name, data_path):
        self._name = name 
        self._cache_path = 'data/cache'
        self._data_path = data_path
        self._annotation_file = os.path.join(self._data_path, self._name + '.txt')
        print 'annotation_file: ', self._annotation_file

        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor',
                         'front-face', 'left-face', 'right-face',
                         'front-head', 'back-head', 'other-face')
        self._class_id_table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2]
        self._num_classes = 3

        #self._class_id_table = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        #        1, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3]
        #self._num_classes = 4

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._num_images = 0
    
    # Convert the class id in voc_ped file to 0-indexed class id.
    def convert_class_id(self, class_id):
        return self._class_id_table[class_id]

    def load_voc_ped_annotation(self):
        """
        Load image and bounding boxes info from overall text file in the
        VOC Ped format.
        """
        assert os.path.exists(self._annotation_file), \
                'Path does not exist: {}'.format(self._annotation_file)

        gt_roidb = []
        # Read VOC Pedestrian annotation file line by line (1 line per image)
        with open(self._annotation_file, "r") as f:
            for line in f:
                # Partition on first space
                s = line.partition(" ");
                # Absoute image path
                image_path = s[0];
                # Annotation: id1 x1 y1 w1 h1 id2 x2 y2 w2 h2 ...
                annot_str = s[2].strip();
         
                values = annot_str.split(" ")
                annot = []
                for val in values:
                    annot.append(int(val))
                
                num_objs = 0
                inds = []
                i = 0
                while i < (len(annot) / 5):
                    if self.convert_class_id(annot[5*i]) > 0:
                        num_objs = num_objs + 1
                        inds.append(5 * i)
                    i = i + 1
                if num_objs <= 0:
                    continue

                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)
                gt_overlaps = np.zeros((num_objs, self._num_classes), dtype=np.float32)
                # "Seg" area for VOC Pedestrian is just the box area
                seg_areas = np.zeros((num_objs), dtype=np.float32)

                sz = PIL.Image.open(image_path).size
                width = sz[0]
                height = sz[1]

                for i in xrange(num_objs):
                    ix = inds[i]
                    cls = self.convert_class_id(annot[ix])
                    gt_classes[i] = cls
                    x1 = float(annot[ix + 1])
                    y1 = float(annot[ix + 2])
                    x2 = float(x1 + annot[ix + 3] - 1)
                    y2 = float(y1 + annot[ix + 4] - 1)
                    x1 = min(max(0, x1), width -1)
                    x2 = min(max(0, x2), width -1)
                    y1 = min(max(0, y1), height -1)
                    y2 = min(max(0, y2), height -1)

                    boxes[i, :] = [x1, y1, x2, y2]
                    gt_overlaps[i, cls] = 1.0
                    seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)

                gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)
                gt_overlaps = gt_overlaps.toarray()
                # need gt_overlaps as a dense array for argmax
                # max overlap with gt over classes (columns)
                max_overlaps = gt_overlaps.max(axis=1)
                # gt class that had the max overlap
                max_classes = gt_overlaps.argmax(axis=1)


                # sanity checks
                # max overlap of 0 => class should be zero (background)
                zero_inds = np.where(max_overlaps == 0)[0]
                assert all(max_classes[zero_inds] == 0)
                # max overlap > 0 => class should not be zero (must be a fg class)
                nonzero_inds = np.where(max_overlaps > 0)[0]
                assert all(max_classes[nonzero_inds] != 0)
                gt_roidb.append({'boxes': boxes, 'gt_overlaps': gt_overlaps,
                    'gt_classes': gt_classes, 'flipped': False, 'image':
                    image_path, 'seg_areas': seg_areas, 'width': width,
                    'height': height, 'max_classes': max_classes,
                    'max_overlaps': max_overlaps})
                self._num_images = self._num_images + 1
        return gt_roidb

    def append_flipped_images_voc_ped(self, roidb):
        num_images = len(roidb)
        for i in xrange(num_images):
            boxes = roidb[i]['boxes'].copy()
            width = roidb[i]['width']
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = width - oldx2 - 1
            boxes[:, 2] = width - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            roidb.append({'boxes': boxes, 'gt_overlaps':
                roidb[i]['gt_overlaps'], 'gt_classes': roidb[i]['gt_classes'],
                'flipped': True, 'image': roidb[i]['image'], 'seg_areas':
                roidb[i]['seg_areas'], 'width': roidb[i]['width'],
                'height': roidb[i]['height'], 'max_classes':
                roidb[i]['max_classes'], 'max_overlaps':
                roidb[i]['max_overlaps']})
        self._num_images = len(roidb)


    def load_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self._cache_path, self._name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self._name, cache_file)
            return roidb

        gt_roidb = self.load_voc_ped_annotation()
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

if __name__ == '__main__':
    from datasets.voc_ped import voc_ped
    d = voc_ped('voc_ped', '/home/mrbing/dataset/VOC')
    #res = d.roidb
    roidb = d.load_roidb()
    print len(roidb)

    #from IPython import embed; embed()
