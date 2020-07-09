# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append('../')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.io import image_preprocess
from help_utils.help_utils import draw_box_cv
from help_utils.tools import *
from libs.configs import cfgs
from libs.fast_rcnn import build_fast_rcnn
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn
from tools import restore_model


def get_imgs():
  mkdir(cfgs.INFERENCE_IMAGE_PATH)
  root_dir = cfgs.INFERENCE_IMAGE_PATH
  img_name_list = os.listdir(root_dir)
  if not img_name_list:
    assert 'no test image in {}!'.format(cfgs.INFERENCE_IMAGE_PATH)
  img_list = [cv2.imread(os.path.join(root_dir, img_name))
              for img_name in img_name_list]
  return img_list, img_name_list


def inference(args):
  with tf.Graph().as_default():

    img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

    img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
    img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                      target_shortside_len=cfgs.SHORT_SIDE_LEN,
                                                                      is_resize=True)

    # ***********************************************************************************************
    # *                                         share net                                           *
    # ***********************************************************************************************
    _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                      inputs=img_batch,
                                      num_classes=None,
                                      is_training=True,
                                      output_stride=None,
                                      global_pool=False,
                                      spatial_squeeze=False)
    # ***********************************************************************************************
    # *                                            RPN                                              *
    # ***********************************************************************************************
    rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                        inputs=img_batch,
                        gtboxes_and_label=None,
                        is_training=False,
                        share_head=cfgs.SHARE_HEAD,
                        share_net=share_net,
                        stride=cfgs.STRIDE,
                        anchor_ratios=cfgs.ANCHOR_RATIOS,
                        anchor_scales=cfgs.ANCHOR_SCALES,
                        scale_factors=cfgs.SCALE_FACTORS,
                        base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                        level=cfgs.LEVEL,
                        top_k_nms=cfgs.RPN_TOP_K_NMS,
                        rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                        max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                        rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                        rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                        rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                        rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                        remove_outside_anchors=False,  # whether remove anchors outside
                        rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

    # rpn predict proposals
    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

    # ***********************************************************************************************
    # *                                         Fast RCNN                                           *
    # ***********************************************************************************************
    fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=img_batch,
                                         feature_pyramid=rpn.feature_pyramid,
                                         rpn_proposals_boxes=rpn_proposals_boxes,
                                         rpn_proposals_scores=rpn_proposals_scores,
                                         img_shape=tf.shape(img_batch),
                                         roi_size=cfgs.ROI_SIZE,
                                         scale_factors=cfgs.SCALE_FACTORS,
                                         roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                         gtboxes_and_label=None,
                                         fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                         fast_rcnn_maximum_boxes_per_img=100,
                                         fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                         show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                         # show detections which score >= 0.6
                                         num_classes=cfgs.CLASS_NUM,
                                         fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                         fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                         fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                         use_dropout=False,
                                         weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                         is_training=False,
                                         level=cfgs.LEVEL)

    fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
        fast_rcnn.fast_rcnn_predict()

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = restore_model.get_restorer(checkpoint_path=args.weights)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      if not restorer is None:
        restorer.restore(sess, restore_ckpt)
        print('restore model')

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess, coord)

      imgs, img_names = get_imgs()
      for i, img in enumerate(imgs):

        start = time.time()

        _img_batch, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category = \
            sess.run([img_batch, fast_rcnn_decode_boxes, fast_rcnn_score, detection_category],
                     feed_dict={img_plac: img})
        end = time.time()

        img_np = np.squeeze(_img_batch, axis=0)

        img_np = draw_box_cv(img_np,
                             boxes=_fast_rcnn_decode_boxes,
                             labels=_detection_category,
                             scores=_fast_rcnn_score)
        mkdir(cfgs.INFERENCE_SAVE_PATH)
        cv2.imwrite(cfgs.INFERENCE_SAVE_PATH + '/{}_fpn.jpg'.format(img_names[i]), img_np)
        view_bar('{} cost {}s'.format(img_names[i], (end - start)), i + 1, len(imgs))

      coord.request_stop()
      coord.join(threads)


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Inference using trained FPN model.')
  parser.add_argument('--weights', dest='weights',
                      help='model path',
                      type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = parse_args()
  print('Called with args:')
  print(args)
  inference(args)
