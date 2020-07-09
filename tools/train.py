#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_network_byname
from libs.configs import cfgs
from libs.rpn import build_rpn
from libs.fpn import build_fpn
from libs.fast_rcnn import build_fast_rcnn
from libs.box_utils.boxes_utils import print_tensors
from libs.box_utils.boxes_utils import draw_boxes_with_categories_and_scores,draw_boxes_with_scores
from tools import restore_model



def train():
    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            img_name_batch, img_batch, gtboxes_and_label_batch = \
                next_batch(data_tfrecord_location=cfgs.DATASET_DIR,
                           batch_size=cfgs.BATCH_SIZE,
                           shortside_len=cfgs.SHORT_SIDE_LEN,
                           longside_len=cfgs.LONG_SIDE_LEN,
                           is_training=True)

            image_height,image_width = tf.shape(img_batch)[1],tf.shape(img_batch)[2]

        # with tf.name_scope('draw_gtboxes'):
        #     gtboxes_in_img = draw_box_with_color(img_batch, tf.reshape(gtboxes_and_label_batch, [-1, 5])[:, :-1],
        #                                          text=tf.shape(gtboxes_and_label_batch)[1])

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


        # fpn
        feature_pyramid = build_fpn.build_feature_pyramid(share_net)  # [P2,P3,P4,P5,P6]
        # ***********************************************************************************************
        # *                                            rpn                                              *
        # ***********************************************************************************************

        rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
                            image_height=image_height,
                            image_width=image_width,
                            gtboxes_and_label=gtboxes_and_label_batch,
                            is_training=True)

        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(is_training=True)
        rpn_location_loss, rpn_classification_loss = rpn.rpn_losses()

        rpn_total_loss = rpn_classification_loss + rpn_location_loss

        # ***********************************************************************************************
        # *                                   Fast RCNN Head                                          *
        # ***********************************************************************************************

        fast_rcnn = build_fast_rcnn.FAST_RCNN(
            feature_pyramid=feature_pyramid,
            rpn_proposals_boxes=rpn_proposals_boxes,
            gtboxes_and_label=gtboxes_and_label_batch,
            origin_image=img_batch,
            is_training=True,
            image_height=image_height,
            image_width=image_width)

        detections = fast_rcnn.head_detection()

        if cfgs.DEBUG:
            print_tensors(rpn_proposals_scores[0, :50], "scores")   # 前50
            print_tensors(rpn_proposals_boxes[0, :50, :], "bbox")
            rpn_proposals_vision = draw_boxes_with_scores(img_batch[0, :, :, :],
                                                          rpn_proposals_boxes[0, :50, :],
                                                          rpn_proposals_scores[0, :50])
            fast_rcnn_vision = draw_boxes_with_categories_and_scores(img_batch[0, :, :, :],
                                                                detections[0, :, :4],
                                                                detections[0, :, 4],
                                                                detections[0, :, 5],
                                                                cfgs.LABEL_TO_NAME)
            tf.summary.image("rpn_proposals_vision", rpn_proposals_vision)
            tf.summary.image("fast_rcnn_vision", fast_rcnn_vision)

        fast_rcnn_location_loss, fast_rcnn_classification_loss = fast_rcnn.head_loss()
        fast_rcnn_total_loss = fast_rcnn_location_loss + fast_rcnn_classification_loss

        # train
        with tf.name_scope("regularization_losses"):
            regularization_list = [tf.nn.l2_loss(w.read_value()) *
                                   cfgs.WEIGHT_DECAY[cfgs.NET_NAME] / tf.cast(tf.size(w.read_value()),
                                                                     tf.float32) for w in tf.trainable_variables() if
                                   'gamma' not in w.name and 'beta' not in w.name]
            regularization_loss = tf.add_n(regularization_list)

        total_loss = regularization_loss + fast_rcnn_total_loss + rpn_total_loss
        total_loss = tf.cond(tf.is_nan(total_loss), lambda: 0.0, lambda: total_loss)

        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.piecewise_constant(global_step,
                                         boundaries=[np.int64(10000), np.int64(20000)],
                                         values=[cfgs.LR, cfgs.LR / 10, cfgs.LR / 100])

        # *                                          Summary                                            *
        # ***********************************************************************************************
        # rpn loss and image
        tf.summary.scalar('rpn_location_loss', rpn_location_loss, family="rpn_loss")
        tf.summary.scalar('rpn_classification_loss', rpn_classification_loss, family="rpn_loss")
        tf.summary.scalar('rpn_total_loss', rpn_total_loss, family="rpn_loss")

        tf.summary.scalar('fast_rcnn_location_loss', fast_rcnn_location_loss, family="head_loss")
        tf.summary.scalar('fast_rcnn_classification_loss', fast_rcnn_classification_loss, family="head_loss")
        tf.summary.scalar('fast_rcnn_total_loss', fast_rcnn_total_loss, family="head_loss")
        tf.summary.scalar("regularization_loss", regularization_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', lr)

        # optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
        #
        # train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)
        with tf.name_scope("optimizer"):
            optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                grads = optimizer.compute_gradients(total_loss)
                # clip gradients
                grads = tf.contrib.training.clip_gradient_norms(grads, cfgs.CLIP_GRADIENT_NORM)
                train_op = optimizer.apply_gradients(grads, global_step)

        summary_op = tf.summary.merge_all()
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        # restorer, restore_ckpt = restore_model.get_restorer(test=False,checkpoint_path=cfgs.TRAINED_MODEL_DIR)
        saver = tf.train.Saver(max_to_keep=3)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # if not restorer is None:
            #     restorer.restore(sess, restore_ckpt)
            #     print('restore model successfully')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            summary_path = cfgs.SUMMARY_PATH
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

            for step in range(cfgs.MAX_ITERATION):
                training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                start = time.time()
                _global_step,_rpn_location_loss, _rpn_classification_loss, \
                _rpn_total_loss, _fast_rcnn_location_loss, _fast_rcnn_classification_loss, \
                _fast_rcnn_total_loss,  _total_loss, _ = \
                    sess.run([global_step,rpn_location_loss, rpn_classification_loss,
                              rpn_total_loss, fast_rcnn_location_loss, fast_rcnn_classification_loss,
                              fast_rcnn_total_loss,total_loss, train_op])
                end = time.time()

                if step % 50 == 0:
                    print("""{}: step{}
                             rpn_loc_loss:{:.4f} | rpn_cla_loss:{:.4f} | rpn_total_loss:{:.4f}
                             fast_rcnn_loc_loss:{:.4f} | fast_rcnn_cla_loss:{:.4f} | fast_rcnn_total_loss:{:.4f}
                             | total_loss:{:.4f} | pre_cost_time:{:.4f}s"""
                          .format(training_time, _global_step, _rpn_location_loss,
                                  _rpn_classification_loss, _rpn_total_loss, _fast_rcnn_location_loss,
                                  _fast_rcnn_classification_loss, _fast_rcnn_total_loss, _total_loss,
                                  (end - start)))


                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, _global_step)
                    summary_writer.flush()

                if (step > 0 and step % 5000 == 0) or (step == cfgs.MAX_ITERATION - 1):
                    save_dir = cfgs.TRAINED_MODEL_DIR
                    save_ckpt = os.path.join(save_dir, 'sarship'+'{}_'.format(
                        cfgs.NET_NAME) + str(_global_step) + 'model.ckpt')
                    saver.save(sess, save_ckpt)
                    print('Weights have been saved to {}.'.format(save_ckpt))


            print('Training done')
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
