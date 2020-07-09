from __future__ import absolute_import, division, print_function

import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(0, '../../')

from libs.networks.slim_nets import (inception_resnet_v2, mobilenet_v1,
                                     resnet_v1, vgg)


def get_network_byname(net_name,
                       inputs,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       spatial_squeeze=True):
    if net_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0001)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                        num_classes=num_classes,
                                                        is_training=is_training,
                                                        global_pool=global_pool,
                                                        output_stride=output_stride,
                                                        spatial_squeeze=spatial_squeeze)
        return logits, end_points

    if net_name == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0001)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_training,
                                                         global_pool=global_pool,
                                                         output_stride=output_stride,
                                                         spatial_squeeze=spatial_squeeze
                                                         )
        return logits, end_points


