# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import colorsys


def clip_boxes_to_img_boundaries(decode_boxes, img_height,img_img_width):
    '''

    :param decode_boxes:
    :return: decode boxes, and already clip to boundaries
    '''

    # xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(decode_boxes))
    with tf.name_scope('clip_boxes_to_img_boundaries'):

        ymin, xmin, ymax, xmax = tf.unstack(decode_boxes, axis=1)
        img_h, img_w = img_height, img_img_width

        xmin = tf.maximum(xmin, 0.0)
        xmin = tf.minimum(xmin, tf.cast(img_w, tf.float32))

        ymin = tf.maximum(ymin, 0.0)
        ymin = tf.minimum(ymin, tf.cast(img_h, tf.float32))  # avoid xmin > img_w, ymin > img_h

        xmax = tf.minimum(xmax, tf.cast(img_w, tf.float32))
        ymax = tf.minimum(ymax, tf.cast(img_h, tf.float32))

        return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def filter_outside_boxes(boxes, img_w, img_h):
    '''
    :param anchors:boxes with format [xmin, ymin, xmax, ymax]
    :param img_h: height of image
    :param img_w: width of image
    :return: indices of anchors that not outside the image boundary
    '''

    with tf.name_scope('filter_outside_boxes'):

        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        xmin_index = tf.greater_equal(xmin, 0)
        ymin_index = tf.greater_equal(ymin, 0)
        xmax_index = tf.less_equal(xmax, img_w)
        ymax_index = tf.less_equal(ymax, img_h)

        indices = tf.transpose(tf.stack([ymin_index, xmin_index, ymax_index, xmax_index]))
        indices = tf.cast(indices, dtype=tf.int32)
        indices = tf.reduce_sum(indices, axis=1)
        indices = tf.where(tf.equal(indices, tf.shape(boxes)[1]))

        return tf.reshape(indices, [-1, ])


def nms_boxes(decode_boxes, scores, iou_threshold, max_output_size, name):
    '''
    1) NMS
    2) get maximum num of proposals
    :return: valid_indices
    '''

    # xmin, ymin, xmax, ymax = tf.unstack(tf.transpose(decode_boxes))
    # ymin, xmin, ymax, xmax = tf.unstack(decode_boxes, axis=1)
    valid_index = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        name=name
    )

    return valid_index


def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):

    '''
    num of boxes less than max num of boxes, so it need to pad with zeros[0, 0, 0, 0]
    :param boxes:
    :param scores: [-1]
    :param max_num_of_boxes:
    :return:
    '''

    pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

    zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
    zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)

    final_boxes = tf.concat([boxes, zero_boxes], axis=0)

    final_scores = tf.concat([scores, zero_scores], axis=0)

    return final_boxes, final_scores




def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result



def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros



def print_tensors(tensor, tensor_name):

    def np_print(ary):
        ary = ary + np.zeros_like(ary)
        print(tensor_name + ':', ary)

        print('shape is: ',ary.shape)
        print(10*"%%%%%")
        return ary
    result = tf.py_func(np_print,
                        [tensor],
                        [tensor.dtype])
    result = tf.reshape(result, tf.shape(tensor))
    result = tf.cast(result, tf.float32)
    sum_ = tf.reduce_sum(result)
    tf.summary.scalar('print_s/{}'.format(tensor_name), sum_)



def draw_boxes_with_scores(img_batch, boxes, scores):

    def draw_box_cv(img, boxes, scores):
        boxes = boxes.astype(np.int64)
        img = np.array(img, np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            score = scores[i]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax), color=color,
                          thickness=2)

            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmin+20, ymin+10),
                          color=color,
                          thickness=-1)

            cv2.putText(img,
                        text=str(np.round(score, 2)),
                        org=(xmin, ymin+10),
                        fontFace=1,
                        fontScale=1,
                        thickness=2,
                        color=(color[1], color[2], color[0]))
            num_of_object += 1
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        return img
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_batch, boxes, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)
    return img_tensor_with_boxes







def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores, label_to_name):

    def draw_box_cv(img, boxes, labels, scores):
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img, np.uint8)
        color = class_colors(labels, 12)

        num_of_object = 0
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmax, ymax),
                              color=color[i],
                              thickness=2)
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin+120, ymin+15),
                              color=color[i],
                              thickness=-1)
                category = label_to_name[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(xmin, ymin+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(255, 255, 255))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        return img

    img_tensor = img_batch
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)
    return img_tensor_with_boxes


def class_colors(class_ids, num_classes, bright=True):
    """
    based on the class id to choose a centrial color to show them
    """
    brightness = 1.0 if bright else 0.7
    colors = np.zeros_like(class_ids)
    hsv = [(i / np.float(num_classes), 1, brightness) for i in range(num_classes)]
    color_map = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    color_map = np.ceil(np.array(color_map) * 255)
    colors = color_map[class_ids - 1]

    return colors