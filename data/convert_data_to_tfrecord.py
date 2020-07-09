# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import sys

sys.path.append('../../')

import xml.etree.cElementTree as ET

import numpy as np

import tensorflow as tf

import glob

import cv2

from help_utils.tools import *

from libs.label_name_dict.label_dict import *

import os

VOC_dir = 'F:\SAR/test_images_complex/'

annotations_dir = 'Annotations'

img_dir = 'JPEGImages'

save_name = 'test_complex_1'

save_dir = 'E:\Tree\dataset/'

img_format = '.jpg'

dataset = 'sarship'



# FLAGS = tf.app.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
    '''

    :param xml_path:
    :return: [num of gtboxes, 5]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_boxes_label = []

    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)

    for obj in root.findall('object'):
        label = obj.find('name').text
        label = NAME_LABEL_MAP[label]

        bbox = obj.find('bndbox')
        ymin = int(float(bbox.find('ymin').text))
        xmin = int(float(bbox.find('xmin').text))
        ymax = int(float(bbox.find('ymax').text))
        xmax = int(float(bbox.find('xmax').text))

        temp_box = [ymin, xmin, ymax, xmax, label]
        gt_boxes_label.append(temp_box)

    return gt_boxes_label




def convert_pascal_to_tfrecord():
    save_path = save_dir + dataset + '_' + save_name + '.tfrecord'

    mkdir(save_dir)

    label_dir = VOC_dir + annotations_dir

    image_dir = VOC_dir + img_dir

    writer = tf.python_io.TFRecordWriter(path=save_path)

    for count, fn in enumerate(os.listdir(image_dir)):
        print(count + 1)

        image_fp = os.path.join(image_dir, fn)

        image_fp = image_fp.replace('\\', '/')

        label_fp = os.path.join(label_dir, fn.replace('.jpg', '.xml'))

        # print('label_fp:',label_fp)

        img_name = str.encode(fn)

        if not os.path.exists(label_fp):
            print('{} is not exist!'.format(label_fp))

            continue

        # img = np.array(Image.open(img_path))

        img = cv2.imread(image_fp)

        sizeImg = img.shape


        img_height = sizeImg[0]

        img_width = sizeImg[1]


        gtbox_label = read_xml_gtbox_and_label(label_fp)



        gtbox_label = np.array(gtbox_label, dtype=np.int32)  # [y1, x1. y2, x2, label]

        if gtbox_label.shape[0] == 0:
            continue

        ymin, xmin, ymax, xmax, label = gtbox_label[:, 0], gtbox_label[:, 1], gtbox_label[:, 2], \
                                        gtbox_label[:,3], gtbox_label[:,4]

        gtbox_label = np.transpose(

            np.stack([ymin, xmin, ymax, xmax, label], axis=0))  # [ymin, xmin, ymax, xmax, label]

        feature = tf.train.Features(feature={

            # maybe do not need encode() in linux

            'img_name': _bytes_feature(img_name),

            'img_height': _int64_feature(img_height),

            'img_width': _int64_feature(img_width),

            'img': _bytes_feature(img.tostring()),

            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),

            'num_objects': _int64_feature(gtbox_label.shape[0])

        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

    # view_bar('Conversion progress', count + 1, len(glob.glob(image_dir + '/*.jpg')))

print('\nConversion is complete!')


if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'

    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
