# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.losses import losses
from libs.box_utils import encode_and_decode, boxes_utils
from libs.box_utils.show_box_in_tensor import draw_boxes_with_categories

from libs.configs import cfgs
from libs.box_utils import iou


class FAST_RCNN(object):
    def __init__(self,
                 feature_pyramid,
                 rpn_proposals_boxes,
                 gtboxes_and_label,  # [batch_size, M, 5]
                 origin_image,
                 is_training,
                 image_height,
                 image_width):

        self.feature_pyramid = feature_pyramid
        self.rpn_proposals_boxes = rpn_proposals_boxes  # [batch_size, N, 4]
        self.gtboxes_and_label = gtboxes_and_label
        self.origin_image = origin_image
        self.IS_TRAINING = is_training
        self.image_height = image_height
        self.image_width = image_width

        self.level = cfgs.LEVEL
        self.min_level = int(self.level[0][1])
        self.max_level = min(int(self.level[-1][1]), 5)




    def assign_level(self, minibatch_reference_proboxes):

        """
        compute the level of rpn_proposals_boxes
        :param: minibatch_reference_proboxes (batch_size, num_proposals, 4)[y1, x1, y2, x2]
        return: (batch_size, num_proposals)
        Note that we have not trim the elements padding is 0 which does not affect the finial result.
        """
        with tf.name_scope('assign_levels'):
            ymin, xmin, ymax, xmax = tf.unstack(minibatch_reference_proboxes, axis=2)

            w = tf.maximum(xmax - xmin, 0.)  # avoid w is negative
            h = tf.maximum(ymax - ymin, 0.)  # avoid h is negative

            levels = tf.round(4. + tf.log(tf.sqrt(w*h + 1e-8)/224.0) / tf.log(2.))  # 4 + log_2(***)

            levels = tf.maximum(levels, tf.ones_like(levels) * (np.float32(self.min_level)))  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * (np.float32(self.max_level)))  # level maximum is 5

            return tf.cast(levels, tf.int32)



    def get_rois_feature(self, proposal_bbox):

        """
        1)get roi from feature map
        2)roi align or roi pooling. Here is roi align
        :param: proposal_bbox: (batch_size, num_proposal, 4)[y1, x1, y2, x2]
        :return:
        all_level_rois: [batch_size, num_proposal, 7, 7, C]
        """

        levels = self.assign_level(proposal_bbox)  # (batch_size, num_proposals)
        with tf.name_scope('obtain_roi_feature'):
            pooled = []
            # this is aimed at reorder the pooling map (batch_size, num_proposal)
            box_to_level = []
            for i in range(self.min_level, self.max_level + 1):  # remove P6
                ix = tf.where(tf.equal(levels, i))
                level_i_proposals = tf.gather_nd(proposal_bbox, ix)

                # Box indicies for crop_and_resize.
                box_indices = tf.cast(ix[:, 0], tf.int32)

                box_to_level.append(ix)

                level_i_proposals = tf.stop_gradient(level_i_proposals)
                box_indices = tf.stop_gradient(box_indices)

                # image_shape = tf.constant([self.image_height-1, self.image_width-1,
                #                            self.image_height-1, self.image_width-1], dtype=tf.float32)
                # normal_level_i_proposals = level_i_proposals / image_shape
                ymin, xmin, ymax, xmax = tf.unstack(level_i_proposals, axis=-1)
                image_height, image_width = tf.cast(self.image_height,tf.float32),tf.cast(self.image_width,tf.float32)
                normalize_ymin = ymin / (image_height - 1)
                normalize_xmin = xmin / (image_width - 1)
                normalize_ymax = ymax / (image_height - 1)
                normalize_xmax = xmax / (image_width - 1)

                normal_level_i_proposals = tf.stack([normalize_ymin,normalize_xmin,normalize_ymax,normalize_xmax],axis=-1)
                level_i_cropped_rois = tf.image.crop_and_resize(self.feature_pyramid['P%d' % i],
                                                                boxes=normal_level_i_proposals,
                                                                box_ind=box_indices,
                                                                crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE])
                pooled.append(level_i_cropped_rois)

            # Pack pooled features into one tensor
            pooled = tf.concat(pooled, axis=0)
            box_to_level = tf.concat(box_to_level, axis=0)
            box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
            box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                     axis=1)

            # Rearrange pooled features to match the order of the original boxes
            # Sort box_to_level by batch then box index
            # TF doesn't have a way to sort by two columns, so merge them and sort.   这块代码没看懂？？?
            sorting_tensor = box_to_level[:, 0] * 10000 + box_to_level[:, 1]
            ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
                box_to_level)[0]).indices[::-1]
            ix = tf.gather(box_to_level[:, 2], ix)
            pooled = tf.gather(pooled, ix)
            reshape_pooled = self.div_batch_and_bboxes_dims([pooled])
            return reshape_pooled



    def div_batch_and_bboxes_dims(self, inputs):
        """
        :param inputs:list of tensor
        :return: list of tensor
        """
        outputs = []
        for input in inputs:
            input_shape = input.get_shape().as_list()
            output = tf.reshape(input, [cfgs.BATCH_SIZE, -1,] + input_shape[1:])
            outputs.append(output)
        if len(inputs) == 1:
            return outputs[0]
        return outputs




    def head_net(self, features, is_training):
        """
        base the feature to compute the finial bbox and scores
        :param features:(batch_size, num_proposal, 7, 7, channels)
        :param is_training: whether change the mean and variance in batch_normalization
        :return:
        fast_rcnn_encode_boxes: (batch_size, num_proposal, num_classes*4)
        fast_rcnn_scores:(batch_size, num_proposal, num_classes)
        """

        def batch_slice_head_net(features, is_training):

            with tf.variable_scope('head_net', reuse=tf.AUTO_REUSE):

                net = slim.conv2d(inputs=features,
                                  num_outputs=1024,
                                  kernel_size=[cfgs.ROI_SIZE,cfgs.ROI_SIZE],
                                  padding='valid',
                                  scope='fast_rcnn_fc1')

                net = slim.batch_norm(inputs=net,
                                      epsilon=0.00001,
                                      is_training=is_training,
                                      fused=True)
                net = tf.nn.relu(net)

                net = slim.conv2d(inputs=net,
                                  num_outputs=1024,
                                  kernel_size=(1,1),
                                  padding='valid',
                                  scope='fast_rcnn_fc2')

                net = slim.batch_norm(inputs=net,
                                      epsilon=0.00001,
                                      is_training=is_training,
                                      fused=True)
                net = tf.nn.relu(net)
                net = tf.squeeze(net, axis=[1, 2])  # [num_of_proposals, 1024]

                fast_rcnn_encode_scores = slim.fully_connected(inputs=net,
                                                        num_outputs=2,
                                                        scope='fast_rcnn_classifier')


                fast_rcnn_encode_boxes = slim.fully_connected(inputs=net,
                                                        num_outputs=2*4,
                                                        scope='fast_rcnn_regressor')

                fast_rcnn_encode_boxes = tf.reshape(fast_rcnn_encode_boxes,[-1,cfgs.NUM_CLASS,4])

                return fast_rcnn_encode_boxes,fast_rcnn_encode_scores

        head_encode_boxes, head_scores =\
            boxes_utils.batch_slice([features],
                                    lambda x: batch_slice_head_net(x,is_training),
                                    cfgs.BATCH_SIZE)

        return head_encode_boxes, head_scores




    def build_head_train_sample(self):

        """
        when training, we should know each reference box's label and gtbox,
        in second stage
        iou >= 0.5 is object
        iou < 0.5 is background
        this function need batch_slice
        :return:
        minibatch_reference_proboxes: (batch_szie, config.HEAD_MINIBATCH_SIZE, 4)[y1, x1, y2, x2]
        minibatch_encode_gtboxes:(batch_szie, config.HEAD_MINIBATCH_SIZE, 4)[dy, dx, log(dh), log(dw)]
        object_mask:(batch_szie, config.HEAD_MINIBATCH_SIZE) 1 indicate is object, 0 indicate is not objects
        label: (batch_szie, config.HEAD_MINIBATCH_SIZE)  # config.HEAD_MINIBATCH_SIZE 表示 classes_id
        """

        with tf.name_scope('build_head_train_sample'):

            def batch_slice_build_sample(gtboxes_and_label, rpn_proposals_boxes):

                with tf.name_scope('select_pos_neg_samples'):
                    gtboxes = tf.cast(
                        tf.reshape(gtboxes_and_label[:, :-1], [-1, 4]), tf.float32)
                    gt_class_ids = tf.cast(
                        tf.reshape(gtboxes_and_label[:, -1], [-1, ]), tf.int32)
                    gtboxes, non_zeros = boxes_utils.trim_zeros_graph(gtboxes, name="trim_gt_box")  # [M, 4]
                    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)
                    rpn_proposals_boxes, _ = boxes_utils.trim_zeros_graph(rpn_proposals_boxes,
                                                                          name="trim_rpn_proposal_train")

                    ious = iou.iou_calculate(rpn_proposals_boxes, gtboxes)  # [N, M]
                    matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)  # [N, ]
                    max_iou_each_row = tf.reduce_max(ious, axis=1)
                    positives = tf.cast(tf.greater_equal(max_iou_each_row, cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD), tf.int32)

                    reference_boxes_mattached_gtboxes = tf.gather(gtboxes, matchs)  # [N, 4]
                    gt_class_ids = tf.gather(gt_class_ids, matchs)  # [N, ]
                    object_mask = tf.cast(positives, tf.float32)  # [N, ]
                    # when box is background, not caculate gradient, so give a weight 0 to avoid caculate gradient
                    gt_class_ids = gt_class_ids * positives

                with tf.name_scope('head_train_minibatch'):
                    # choose the positive indices
                    positive_indices = tf.reshape(tf.where(tf.equal(object_mask, 1.)), [-1])
                    num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                                  tf.cast(cfgs.FAST_RCNN_MINIBATCH_SIZE*cfgs.FAST_RCNN_POSITIVE_RATE,
                                                          tf.int32))
                    positive_indices = tf.random_shuffle(positive_indices)
                    positive_indices = tf.slice(positive_indices, begin=[0], size=[num_of_positives])
                    # choose the negative indices,
                    # Strictly propose the proportion of positive and negative is 1:3
                    negative_indices = tf.reshape(tf.where(tf.equal(object_mask, 0.)), [-1])
                    num_of_negatives = tf.cast(int(1. / cfgs.FAST_RCNN_POSITIVE_RATE) * num_of_positives, tf.int32)\
                                       - num_of_positives

                    num_of_negatives = tf.minimum(tf.shape(negative_indices)[0], num_of_negatives)
                    negative_indices = tf.random_shuffle(negative_indices)
                    negative_indices = tf.slice(negative_indices, begin=[0], size=[num_of_negatives])

                    minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
                    minibatch_reference_gtboxes = tf.gather(reference_boxes_mattached_gtboxes,
                                                            minibatch_indices)
                    minibatch_reference_proboxes = tf.gather(rpn_proposals_boxes, minibatch_indices)
                    # encode gtboxes
                    minibatch_encode_gtboxes = \
                        encode_and_decode.encode_boxes(
                            unencode_boxes=minibatch_reference_gtboxes,
                            reference_boxes=minibatch_reference_proboxes,
                            scale_factors=cfgs.BBOX_STD_DEV)
                    object_mask = tf.gather(object_mask, minibatch_indices)
                    gt_class_ids = tf.gather(gt_class_ids, minibatch_indices)

                    # padding if necessary
                    gap = tf.cast(cfgs.FAST_RCNN_MINIBATCH_SIZE - (num_of_positives + num_of_negatives), dtype=tf.int32)
                    bbox_padding = tf.zeros((gap, 4))
                    minibatch_reference_proboxes = tf.concat([minibatch_reference_proboxes, bbox_padding], axis=0)
                    minibatch_encode_gtboxes = tf.concat([minibatch_encode_gtboxes, bbox_padding], axis=0)
                    object_mask = tf.pad(object_mask, [(0, gap)])
                    gt_class_ids = tf.pad(gt_class_ids, [(0, gap)])

                return minibatch_reference_proboxes, minibatch_encode_gtboxes, object_mask, gt_class_ids

            minibatch_reference_proboxes, minibatch_encode_gtboxes, object_mask, gt_class_ids = \
                    boxes_utils.batch_slice([self.gtboxes_and_label, self.rpn_proposals_boxes],
                                            lambda x, y: batch_slice_build_sample(x, y),
                                            cfgs.BATCH_SIZE)
        if cfgs.DEBUG:
            gt_vision = draw_boxes_with_categories(self.origin_image[0],
                                                   self.gtboxes_and_label[0, :, :4],
                                                   self.gtboxes_and_label[0, :, 4],
                                                   cfgs.LABEL_TO_NAME)
            tf.summary.image("gt_vision", gt_vision)

            draw_bbox_train = draw_boxes_with_categories(self.origin_image[0],
                                                         minibatch_reference_proboxes[0],
                                                         gt_class_ids[0],
                                                         cfgs.LABEL_TO_NAME)
            tf.summary.image("train_proposal", draw_bbox_train)

        return minibatch_reference_proboxes, minibatch_encode_gtboxes, object_mask, gt_class_ids



    def head_loss(self):

        minibatch_reference_proboxes, minibatch_encode_gtboxes,\
        object_mask, gt_class_ids = self.build_head_train_sample()

        pooled_feature = self.get_rois_feature(minibatch_reference_proboxes)

        fast_rcnn_predict_boxes, fast_rcnn_predict_scores = self.head_net(pooled_feature, self.IS_TRAINING)

        with tf.variable_scope("head_loss"):
            # from fast_rcnn_predict_boxes choose corresponding encode
            row_index = tf.range(0, tf.shape(gt_class_ids)[1])
            row_index = tf.expand_dims(row_index, 0)
            multi_row_index = tf.tile(row_index, [cfgs.BATCH_SIZE, 1])
            multi_row_index = tf.expand_dims(multi_row_index, axis=-1)
            expand_gt_class_ids = tf.expand_dims(gt_class_ids, axis=-1)
            index = tf.concat([multi_row_index, expand_gt_class_ids], axis=-1)  # 匹配对应gt那个类别的预测框吗？
            fast_rcnn_predict_boxes = boxes_utils.batch_slice([fast_rcnn_predict_boxes, index],
                                                              lambda x, y: tf.gather_nd(x, y),
                                                              cfgs.BATCH_SIZE)

            # loss
            with tf.variable_scope('head_class_loss'):
                fast_rcnn_classification_loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_class_ids,  # [batch_size,num]
                                                                                       logits=fast_rcnn_predict_scores) # [batch_size,num,2]

                fast_rcnn_classification_loss = tf.cond(tf.is_nan(fast_rcnn_classification_loss), lambda: 0.0,
                                                        lambda: fast_rcnn_classification_loss)

            with tf.variable_scope('head_location_loss'):
                fast_rcnn_location_loss = losses.l1_smooth_losses(predict_boxes=fast_rcnn_predict_boxes,
                                                                  gtboxes=minibatch_encode_gtboxes,
                                                                  object_weights=object_mask)

            return fast_rcnn_location_loss, fast_rcnn_classification_loss



    def head_detection(self):
        """
        compute the predict bboxes, categories, categories
        :return:
        fast_rcnn_categories_bboxs:(batch_size, num_proposals, 4)
        fast_rcnn_categories_scores:(batch_size, num_propsals)
        fast_rcnn_categories:(batch_size, num_propsals)
        """


        # (batch_size, num_proposal, 7, 7, channels)
        pooled_feature = self.get_rois_feature(self.rpn_proposals_boxes)
        head_predict_boxes, head_predict_scores = self.head_net(pooled_feature, False)

        with tf.name_scope("head_detection"):

            head_softmax_scores = slim.softmax(head_predict_scores)  # [N, -1, num_classes]
            # gain the highest category and score and bounding box
            head_categories = tf.argmax(head_softmax_scores, axis=2, output_type=tf.int32) # (N, -1)
            row_index = tf.range(0, tf.shape(head_categories)[1])
            row_index = tf.expand_dims(row_index, 0)
            multi_row_index = tf.tile(row_index, [cfgs.BATCH_SIZE, 1])
            multi_row_index = tf.expand_dims(multi_row_index, axis=-1)
            expand_head_categories = tf.expand_dims(head_categories, axis=-1)
            index = tf.concat([multi_row_index, expand_head_categories], axis=-1)
            head_categories_bboxs = boxes_utils.batch_slice([head_predict_boxes, index],  # 只取类别概率最大的那个box
                                                                 lambda x, y: tf.gather_nd(x, y),
                                                                 cfgs.BATCH_SIZE)
            head_categories_scores = tf.reduce_max(head_softmax_scores, axis=2, keepdims=False)# (N, -1)  # 那个Box的分数

            detections = self.head_proposals(self.rpn_proposals_boxes,  # decode_boxes
                                             head_categories_bboxs,  # fast_rcnn_encode_boxes
                                             head_categories,
                                             head_categories_scores,
                                             self.image_height,
                                             self.image_width)

            return detections



    def head_proposals(self, rpn_proposal_bbox, encode_boxes, categories, scores, image_height,image_width):
        """
        padding zeros to keep alignments
        :return:
        detection_boxes_scores_labels:(batch_size, config.MAX_DETECTION_INSTANCE, 6)
        """

        def batch_slice_head_proposals(rpn_proposal_bbox,
                                       encode_boxes,
                                       categories,
                                       scores,
                                       image_height,
                                       image_width):
            """
            mutilclass NMS
            :param rpn_proposal_bbox: (N, 4)
            :param encode_boxes: (N, 4)
            :param categories:(N, )
            :param scores: (N, )
            :param image_window:(y1, x1, y2, x2) the boundary of image
            :return:
            detection_boxes_scores_labels : (-1, 6)[y1, x1, y2, x2, scores, labels]
            """
            with tf.name_scope('head_proposals'):
                # trim the zero graph
                rpn_proposal_bbox, non_zeros = boxes_utils.trim_zeros_graph(rpn_proposal_bbox,
                                                                            name="trim_proposals_detection")
                encode_boxes = tf.boolean_mask(encode_boxes, non_zeros)
                categories = tf.boolean_mask(categories, non_zeros)
                scores = tf.boolean_mask(scores, non_zeros)
                fast_rcnn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=encode_boxes,
                                                                        reference_boxes=rpn_proposal_bbox,
                                                                        scale_factors=cfgs.BBOX_STD_DEV)
                fast_rcnn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(fast_rcnn_decode_boxes,
                                                                                  image_height,image_width)

                # remove the background
                keep = tf.cast(tf.where(categories > 0)[:, 0], tf.int32)
                if cfgs.DEBUG:
                    print_categories = tf.gather(categories, keep)
                    print_scores = tf.gather(scores, keep)
                    num_item = tf.minimum(tf.shape(print_scores)[0], 100)
                    print_scores_vision, print_index = tf.nn.top_k(print_scores, k=num_item)
                    print_categories_vision = tf.gather(print_categories, print_index)
                    boxes_utils.print_tensors(print_categories_vision, "categories")
                    boxes_utils.print_tensors(print_scores_vision, "scores")
                # Filter out low confidence boxes
                if cfgs.FINAL_SCORE_THRESHOLD:  # 0.7
                    conf_keep = tf.cast(tf.where(scores >=cfgs.FINAL_SCORE_THRESHOLD)[:, 0], tf.int32)
                    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                                    tf.expand_dims(conf_keep, 0))
                    keep = tf.sparse_tensor_to_dense(keep)[0]

                pre_nms_class_ids = tf.gather(categories, keep)
                pre_nms_scores = tf.gather(scores, keep)
                pre_nms_rois = tf.gather(fast_rcnn_decode_boxes, keep)
                unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

                def nms_keep_map(class_id):
                    """Apply Non-Maximum Suppression on ROIs of the given class."""
                    # Indices of ROIs of the given class
                    ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
                    # Apply NMS
                    class_keep = tf.image.non_max_suppression(
                        tf.gather(pre_nms_rois, ixs),
                        tf.gather(pre_nms_scores, ixs),
                        max_output_size=cfgs.DETECTION_MAX_INSTANCES,   # 最多200条
                        iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)  # 0.3 太高就过滤完了
                    # Map indicies
                    class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
                    # Pad with -1 so returned tensors have the same shape
                    gap = cfgs.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
                    class_keep = tf.pad(class_keep, [(0, gap)],
                                        mode='CONSTANT', constant_values=-1)
                    # Set shape so map_fn() can infer result shape
                    class_keep.set_shape([cfgs.DETECTION_MAX_INSTANCES])
                    return class_keep
                # 2. Map over class IDs
                nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                                     dtype=tf.int32)
                # 3. Merge results into one list, and remove -1 padding
                nms_keep = tf.reshape(nms_keep, [-1])
                nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
                # 4. Compute intersection between keep and nms_keep
                keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                                tf.expand_dims(nms_keep, 0))
                keep = tf.sparse_tensor_to_dense(keep)[0]
                # Keep top detections
                roi_count = cfgs.DETECTION_MAX_INSTANCES
                class_scores_keep = tf.gather(scores, keep)
                num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
                top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
                keep = tf.gather(keep, top_ids)

                # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
                # Coordinates are normalized.
                detections = tf.concat([
                    tf.gather(fast_rcnn_decode_boxes, keep),
                    tf.to_float(tf.gather(categories, keep))[..., tf.newaxis],
                    tf.gather(scores, keep)[..., tf.newaxis]
                ], axis=1)

                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = cfgs.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
                detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")

                return detections

        detections = boxes_utils.batch_slice([rpn_proposal_bbox, encode_boxes, categories, scores],
                                             lambda x, y, z, u: batch_slice_head_proposals(x, y, z, u,image_height,image_width),
                                             cfgs.BATCH_SIZE)
        return detections






