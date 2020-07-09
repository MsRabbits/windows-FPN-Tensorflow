#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import tensorflow as tf
from data.io import image_preprocess
from matplotlib import pyplot as plt
import cv2
import numpy as np

from libs.box_utils import draw_boxes_in_image
from libs.configs import cfgs


def read_single_example_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string)})

    image_name = features['img_name']
    image_height = tf.cast(features['img_height'], tf.int32)
    image_width = tf.cast(features['img_width'], tf.int32)
    image_data = tf.decode_raw(features['img'], tf.uint8)
    image = tf.reshape(image_data, shape=[image_height, image_width, 1])
    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

    return image_name, image, gtboxes_and_label



def read_and_preprocess_single_image(filename_queue, shortside_len,longside_len, is_training):
    image_name,image, gt_boxes_and_label = read_single_example_and_decode(filename_queue)
    image = tf.cast(image, tf.float32)
    # image = image - tf.constant([103.939, 116.779, 123.68])


    if is_training:
        image, gt_boxes_and_label =image_preprocess.short_side_resize(image, gt_boxes_and_label, shortside_len,longside_len)
        image, gt_boxes_and_label = image_preprocess.random_flip_left_right(image, gt_boxes_and_label)  # 随机水平翻转

    else:
        image, gt_boxes_and_label = image_preprocess.short_side_resize(image, gt_boxes_and_label, shortside_len,longside_len)

    # image = tf.divide(image, 255.)
    return image_name,image, gt_boxes_and_label



def next_batch(data_tfrecord_location, batch_size, shortside_len,longside_len, is_training):

    pattern = data_tfrecord_location
    filename_tensorlist = tf.train.match_filenames_once(pattern)    # 正则匹配项 返回一个数据集列表 此处应该就一个 如果有多个tfrecord这块就是多个
    filename_queue = tf.train.string_input_producer(filename_tensorlist)  # 创建输入队列

    image_name,image, gt_boxes_and_label = read_and_preprocess_single_image(filename_queue, shortside_len,longside_len,is_training)
    image /= 255.

    image_name_batch, image_batch, gt_boxes_and_label_batch\
        = tf.train.batch(
        [image_name,image, gt_boxes_and_label],
        batch_size=batch_size,
        num_threads=16,      # 线程数
        capacity=500,
        dynamic_pad=True
    )

    return image_name_batch, image_batch, gt_boxes_and_label_batch


data_tfrecord_location = 'F:\Tree\dataset/sarship_train_single_channel.tfrecord'
def ceshi():
    image_name_batch, image_batch, gt_boxes_and_label_batch\
        = next_batch(data_tfrecord_location,batch_size=1,shortside_len=600,longside_len=800,is_training=True)

    # 一个batch中 会根据最多的那个gt_boxes的个数 其他的补零

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in range(5):
                if coord.should_stop():
                    break
                image_name, image, gt_boxes_and_label = sess.run([tf.squeeze(image_name_batch), tf.squeeze(image_batch,axis=0),
                                                                  tf.squeeze(gt_boxes_and_label_batch,axis=0)])
                # plt.imshow(image[0])
                # plt.show()
                print(image_name, image.shape, gt_boxes_and_label)

                gt_boxes = gt_boxes_and_label[:, :4]
                labels = gt_boxes_and_label[:, 4]
                scores = np.ones(shape=np.shape(labels))

                image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                gt_boxes = draw_boxes_in_image.draw_boxes_with_label_and_scores(image, boxes=gt_boxes,
                                                                                labels=labels,
                                                                                scores=scores)

                if not os.path.exists(cfgs.GT_SAVE_PATH):
                    os.makedirs(cfgs.GT_SAVE_PATH)

                cv2.imwrite(cfgs.GT_SAVE_PATH + '/' + str(image_name) + '.jpg',
                            gt_boxes[:, :, ::-1])

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__=='__main__':
    ceshi()


