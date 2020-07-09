# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'sarship':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      "sarship": 1
  }
elif cfgs.DATASET_NAME == 'SSDD':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      "ship": 1
  }
elif cfgs.DATASET_NAME == 'airplane':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      "airplane": 1
  }
elif cfgs.DATASET_NAME == 'nwpu':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      'airplane': 1,
      'ship': 2,
      'storage tank': 3,
      'baseball diamond': 4,
      'tennis court': 5,
      'basketball court': 6,
      'ground track field': 7,
      'harbor': 8,
      'bridge': 9,
      'vehicle': 10,
  }
elif cfgs.DATASET_NAME == 'pascal':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      'aeroplane': 1,
      'bicycle': 2,
      'bird': 3,
      'boat': 4,
      'bottle': 5,
      'bus': 6,
      'car': 7,
      'cat': 8,
      'chair': 9,
      'cow': 10,
      'diningtable': 11,
      'dog': 12,
      'horse': 13,
      'motorbike': 14,
      'person': 15,
      'pottedplant': 16,
      'sheep': 17,
      'sofa': 18,
      'train': 19,
      'tvmonitor': 20
  }
elif cfgs.DATASET_NAME == 'icecream':
  NAME_LABEL_MAP = {}
  NAME_LABEL_MAP['back_ground'] = 0
  with open('classes.txt') as f:
    lines = [line.strip() for line in f.readlines()]
  for i, line in enumerate(lines, 1):
    NAME_LABEL_MAP[line] = i
elif cfgs.DATASET_NAME == 'layer':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      "层": 1
  }
elif cfgs.DATASET_NAME == 'shelf':
  NAME_LABEL_MAP = {
      'back_ground': 0,
      "货架分节": 1
  }
elif cfgs.DATASET_NAME == 'coca':
  NAME_LABEL_MAP = {}
  NAME_LABEL_MAP['back_ground'] = 0
  with open('data/{}/classes.txt'.format(cfgs.DATASET_NAME)) as f:
    lines = [line.strip() for line in f.readlines()]
  for i, line in enumerate(lines, 1):
    NAME_LABEL_MAP[line] = i
elif cfgs.DATASET_NAME == 'cooler':
  NAME_LABEL_MAP = {}
  NAME_LABEL_MAP['back_ground'] = 0
  with open('data/{}/classes.txt'.format(cfgs.DATASET_NAME)) as f:
    lines = [line.strip() for line in f.readlines()]
  for i, line in enumerate(lines, 1):
    NAME_LABEL_MAP[line] = i
else:
  assert 'please set label dict!'


def get_label_name_map():
  reverse_dict = {}
  for name, label in NAME_LABEL_MAP.items():
    reverse_dict[label] = name
  return reverse_dict


LABEl_NAME_MAP = get_label_name_map()
