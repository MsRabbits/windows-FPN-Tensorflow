#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


def max_length_limitation(length, length_limitation):
  return tf.cond(tf.less(length, length_limitation),
                 true_fn=lambda: length,
                 false_fn=lambda: length_limitation)


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len, length_limitation=1200):
  '''

  :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
  :param target_shortside_len:
  :param length_limitation: set max length to avoid OUT OF MEMORY
  :return:
  '''
  img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
  new_h, new_w = tf.cond(tf.less(img_h, img_w),
                         true_fn=lambda: (target_shortside_len,
                                          max_length_limitation(target_shortside_len * img_w // img_h,
                                                                length_limitation)),
                         false_fn=lambda: (
                         max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                         target_shortside_len))

  img_tensor = tf.expand_dims(img_tensor, axis=0)
  img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

  xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)

  new_xmin, new_ymin = xmin * new_w // img_w, ymin * new_h // img_h
  new_xmax, new_ymax = xmax * new_w // img_w, ymax * new_h // img_h
  img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3

  return img_tensor, tf.transpose(tf.stack([new_xmin, new_ymin, new_xmax, new_ymax, label], axis=0))



# def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
#   '''
#
#   :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5]
#   :param target_shortside_len:
#   :return:
#   '''
#
#   h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
#
#   new_h, new_w = tf.cond(tf.less(h, w),
#                          true_fn=lambda: (target_shortside_len, target_shortside_len+100),
#                          false_fn=lambda: (target_shortside_len+100 , target_shortside_len))
#
#   img_tensor = tf.expand_dims(img_tensor, axis=0)
#   img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
#
#
#   ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
#
#   xmin, xmax = xmin * new_w//w, xmax * new_w//w
#   ymin, ymax = ymin * new_h//h, ymax * new_h//h
#
#   img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure imgtensor rank is 3
#
#   return img_tensor, tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label], axis=0))



def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
  h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

  img_tensor = tf.expand_dims(img_tensor, axis=0)
  if is_resize:
    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                           false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

  return img_tensor  # [1, h, w, c]


def random_flip_left_right(img_tensor, gtboxes_and_label):

  h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
  coin = np.random.rand()
  if coin > 0.5:
    img_tensor = tf.image.flip_left_right(img_tensor)

    # xmin, ymin, xmax, ymax = gtboxes_and_label[:, 0], gtboxes_and_label[:, 1], \
    #                          gtboxes_and_label[:, 2], gtboxes_and_label[:, 3]
    #
    # label = gtboxes_and_label[:, 4]
    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_xmin = w - xmax
    new_xmax = w - xmin
    # return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))
    return img_tensor, tf.transpose(tf.stack([ymin, new_xmin, ymax, new_xmax, label], axis=0))
  else:
    return img_tensor,  gtboxes_and_label
