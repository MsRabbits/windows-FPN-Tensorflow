#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def l1_smooth_losses(predict_boxes, gtboxes, object_weights):
    """
    :param predict_boxes: [batch_size, num_boxes, 4]
    :param gtboxes: [batch_size, num_boxes]

    :return:
    """
    # enhanced robustness
    gtboxes = tf.where(tf.is_nan(gtboxes), predict_boxes, gtboxes)
    # choose positive objects

    object_weights = tf.cast(object_weights, tf.int32)
    index = tf.cast(tf.where(tf.equal(object_weights, 1)), tf.int32)
    predict_boxes = tf.gather_nd(predict_boxes, index)
    gtboxes = tf.gather_nd(gtboxes, index)
    diff = predict_boxes - gtboxes
    abs_diff = tf.cast(tf.abs(diff), tf.float32)

    # avoid proposal is no objects
    smooth_box_loss = tf.cond(tf.size(gtboxes) > 0,
                              lambda: tf.reduce_mean(tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(abs_diff), abs_diff - 0.5)),
                              lambda: 0.0)

    return smooth_box_loss
