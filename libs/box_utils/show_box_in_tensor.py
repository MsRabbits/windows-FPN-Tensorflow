from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import colorsys
import cv2



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



def draw_boxes_with_categories(img_batch, boxes, labels, label_to_name):

    def draw_box_cv(img, boxes, labels):
        boxes = boxes.astype(np.int64)
        img = np.array(img, np.uint8)

        num_of_object = 0
        color = class_colors(labels, 12)
        for i, box in enumerate(boxes):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color[i],
                          thickness=2)

            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmin + 20, ymin + 10),
                          color=color[i],
                          thickness=-1)

            category = label_to_name[labels[i]]
            cv2.putText(img,
                        text=category,
                        org=(xmin, ymin + 10),
                        fontFace=1,
                        fontScale=1,
                        thickness=2,
                        color=(255, 255, 255))
            num_of_object += 1
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        return img

    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_batch, boxes, labels],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    img_tensor_with_boxes = tf.expand_dims(img_tensor_with_boxes, 0)
    return img_tensor_with_boxes