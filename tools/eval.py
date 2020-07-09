# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# from __future__ import absolute_import, division, print_function
#
# import os
# import sys
# import time
#
# import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
#
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#
# from data.io.read_tfrecord import next_batch
# from data.io.image_preprocess import short_side_resize_for_inference_data
# from libs.networks.network_factory import get_network_byname
# from libs.configs import cfgs
# from libs.rpn import build_rpn
# from libs.fpn import build_fpn
# from libs.fast_rcnn import build_fast_rcnn
# from libs.box_utils.boxes_utils import print_tensors
# from libs.box_utils.boxes_utils import draw_boxes_with_categories_and_scores,draw_boxes_with_scores
# from tools import restore_model
#
#
#
# def eval():
#     with tf.Graph().as_default():
#         with tf.name_scope('get_batch'):
#             img_name_batch, img_batch, gtboxes_and_label_batch = \
#                 next_batch(data_tfrecord_location=cfgs.DATASET_DIR,
#                            batch_size=cfgs.BATCH_SIZE,
#                            shortside_len=cfgs.SHORT_SIDE_LEN,
#                            is_training=True)
#
#             image_height, image_width = tf.shape(img_batch)[1], tf.shape(img_batch)[2]
#
#
#         # ***********************************************************************************************
#         # *                                         share net                                           *
#         # ***********************************************************************************************
#         _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
#                                           inputs=img_batch,
#                                           num_classes=None,
#                                           is_training=True,
#                                           output_stride=None,
#                                           global_pool=False,
#                                           spatial_squeeze=False)
#
#
#         # fpn
#         feature_pyramid = build_fpn.build_feature_pyramid(share_net)  # [P2,P3,P4,P5,P6]
#         # ***********************************************************************************************
#         # *                                            rpn                                              *
#         # ***********************************************************************************************
#
#         rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
#                             image_height=image_height,
#                             image_width=image_width,
#                             gtboxes_and_label=gtboxes_and_label_batch)
#
#         rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(is_training=True)
#
#         # ***********************************************************************************************
#         # *                                   Fast RCNN Head                                          *
#         # ***********************************************************************************************
#
#         fast_rcnn = build_fast_rcnn.FAST_RCNN(
#             feature_pyramid=feature_pyramid,
#             rpn_proposals_boxes=rpn_proposals_boxes,
#             gtboxes_and_label=gtboxes_and_label_batch,
#             origin_image=img_batch,
#             is_training=True,
#             image_height=image_height,
#             image_width=image_width)
#
#         detections = fast_rcnn.head_detection()
#
#         # 梯度裁剪
#         # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         # with tf.control_dependencies([tf.group(*update_ops)]):
#         #     grads = optimizer.compute_gradients(total_loss)
#         #     # clip gradients
#         #     grads = tf.contrib.training.clip_gradient_norms(grads, net_config.CLIP_GRADIENT_NORM)
#         #     train_op = optimizer.apply_gradients(grads, global_step)
#
#         init_op = tf.group(
#             tf.global_variables_initializer(),
#             tf.local_variables_initializer())
#
#         restorer, restore_ckpt = restore_model.get_restorer(test=True,checkpoint_path=cfgs.TRAINED_MODEL_DIR)
#         saver = tf.train.Saver(max_to_keep=3)
#
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         with tf.Session(config=config) as sess:
#             sess.run(init_op)
#             if not restorer is None:
#                 restorer.restore(sess, restore_ckpt)
#                 print('restore model successfully')
#             img_name_batch,resized_img, detections,gtboxes_and_label_batch = sess.run(img_name_batch,img_batch,detections,
#                                                                                       gtboxes_and_label_batch)
#             img_name = tf.squeeze(img_name_batch)
#             img = tf.squeeze(img_batch)
#             detections = tf.squeeze(img_batch)
#             gtboxes_and_label = tf.squeeze(gtboxes_and_label_batch)
#             boxes_per_img,categories_per_img,scores_per_img = tf.unstack(detections,axis=1)
#
#             gt_boxes = gtboxes_and_label[:,:-1]
#             ymin,xmin,ymax,xmax = boxes_per_img[:,0],boxes_per_img[:,1],boxes_per_img[:,2],boxes_per_img[:,3]
#             gt_ymin,gt_xmin,gt_ymax,gt_xmax = tf.unstack(gt_boxes,axis=-1)
#
#
#
#
#
#
import numpy as np
import os
import tensorflow as tf
import cv2
import time
import math
from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_network_byname
from libs.configs import cfgs
from libs.rpn import build_rpn
from libs.fpn import build_fpn
from libs.fast_rcnn import build_fast_rcnn
from libs.box_utils.boxes_utils import print_tensors
from libs.box_utils.boxes_utils import draw_boxes_with_categories_and_scores,draw_boxes_with_scores
from tools import restore_model
from data.io.image_preprocess import short_side_resize_for_inference_data
import sys
import xml.etree.ElementTree as ET
from libs.label_name_dict import label_dict

NAME_LABEL_MAP = {
    'back_ground': 0,
    "sarship": 1
}


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_with_plac(num_imgs, eval_dir,img_root,showbox,annotation_dir):

    # with open('/home/yjr/DataSet/VOC/VOC_test/VOC2007/ImageSets/Main/aeroplane_test.txt') as f:
    #     all_lines = f.readlines()
    # test_imgname_list = [a_line.split()[0].strip() for a_line in all_lines]

    test_imgname_list = [item for item in os.listdir(eval_dir)
                              if item.endswith(('.jpg', 'jpeg', '.png', '.tif', '.tiff'))]
    if num_imgs == np.inf:
        real_test_imgname_list = test_imgname_list
    else:
        real_test_imgname_list = test_imgname_list[: num_imgs]

    img_plac = tf.placeholder(dtype=tf.float32,shape=[None,None,3])
    img = img_plac - tf.constant([103.939, 116.779, 123.68])
    img_batch = short_side_resize_for_inference_data(img,cfgs.SHORT_SIDE_LEN)
    h,w = img.shape[0],img.shape[1]
    gt_boxes_label = tf.placeholder(dtype=tf.float32,shape=[None,5])
    gt_boxes_label_batch = tf.expand_dims(gt_boxes_label,axis=0)

    image_height, image_width = tf.shape(img_batch)[1], tf.shape(img_batch)[2]

    _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                      inputs=img_batch,
                                      num_classes=None,
                                      is_training=False,
                                      output_stride=None,
                                      global_pool=False,
                                      spatial_squeeze=False)

    feature_pyramid = build_fpn.build_feature_pyramid(share_net)
    rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
                        image_height=image_height,
                        image_width=image_width,
                        gtboxes_and_label=gt_boxes_label_batch,is_training=False)

    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(is_training=False)

    fast_rcnn = build_fast_rcnn.FAST_RCNN(
        feature_pyramid=feature_pyramid,
        rpn_proposals_boxes=rpn_proposals_boxes,
        gtboxes_and_label=gt_boxes_label_batch,
        origin_image=img_batch,
        is_training=True,
        image_height=image_height,
        image_width=image_width)

    detections = fast_rcnn.head_detection()
    detection_boxes, detection_category,detection_scores = tf.squeeze(detections[:,:,:4],axis=0),\
                                                            tf.squeeze(detections[:,:,4],axis=0),\
                                                            tf.squeeze(detections[:,:,5],axis=0)

    indices = tf.reshape(tf.where(tf.greater_equal(detection_scores, cfgs.FINAL_SCORE_THRESHOLD)), [-1])
    detection_boxes = tf.gather(detection_boxes,indices)
    detection_scores = tf.gather(detection_scores, indices)
    detection_category = tf.gather(detection_category, indices)


    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = restore_model.get_restorer(test=True,checkpoint_path=cfgs.chekpoint_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        all_boxes = []
        for i, a_img_name in enumerate(real_test_imgname_list):
            raw_img = cv2.imread(os.path.join(img_root, a_img_name))
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img}  # cv is BGR. But need RGB
                )
            print(a_img_name,detected_boxes,detected_scores,detected_categories)
            end = time.time()
            ymin, xmin, ymax, xmax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                     detected_boxes[:, 2], detected_boxes[:, 3]
            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h

            boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
            dets = np.hstack((detected_categories.reshape(-1, 1),
                              detected_scores.reshape(-1, 1),
                              boxes))
            all_boxes.append(dets)

            view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(real_test_imgname_list))


    voc_evaluate_detections(all_boxes=all_boxes,
                            test_annotation_path=annotation_dir,
                            test_imgid_list=real_test_imgname_list)


def voc_evaluate_detections(all_boxes, test_annotation_path, test_imgid_list):
    '''

    :param all_boxes: is a list. each item reprensent the detections of a img.

    The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
    Note that: if none detections in this img. that the detetions is : []
    :return:
    '''
    test_imgid_list = [item.split('.')[0] for item in test_imgid_list]

    write_voc_results_file(all_boxes, test_imgid_list=test_imgid_list,
                           det_save_dir=os.path.join(cfgs.EVALUATE_DIR, cfgs.NET_NAME))
    do_python_eval(test_imgid_list, test_annotation_path=test_annotation_path)




def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
    '''

    :param all_boxes: is a list. each item reprensent the detections of a img.
    the detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
    Note that: if none detections in this img. that the detetions is : []

    :param test_imgid_list:
    :param det_save_path:
    :return:
    '''
    for cls, cls_id in NAME_LABEL_MAP.items():
        if cls == 'back_ground':
            continue
        print("Writing {} VOC resutls file".format(cls))

        mkdir(det_save_dir)
        det_save_path = os.path.join(det_save_dir, "det_" + cls + ".txt")
        with open(det_save_path, 'wt') as f:
            for index, img_name in enumerate(test_imgid_list):
                this_img_detections = all_boxes[index]

                this_cls_detections = this_img_detections[this_img_detections[:, 0] == cls_id]
                if this_cls_detections.shape[0] == 0:
                    continue  # this cls has none detections in this img
                for a_det in this_cls_detections:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(img_name, a_det[1],
                                   a_det[2], a_det[3],
                                   a_det[4], a_det[5]))  # that is [img_name, score, xmin, ymin, xmax, ymax]

def do_python_eval(test_imgid_list, test_annotation_path):
    AP_list = []
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    # color_list = colors.cnames.keys()[::6]

    for cls, index in NAME_LABEL_MAP.items():
        if cls == 'back_ground':
            continue
        recall, precision, AP = voc_eval(detpath=os.path.join(cfgs.EVALUATE_DIR, cfgs.NET_NAME),
                                         test_imgid_list=test_imgid_list,
                                         cls_name=cls,
                                         annopath=test_annotation_path)
        AP_list += [AP]
        print("cls : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, recall[-1], precision[-1], AP))
        # plt.plot(recall, precision, label=cls, color=color_list[index])
        # plt.legend(loc='upper right')
        print(10 * "__")
    # plt.show()
    # plt.savefig(cfgs.VERSION+'.jpg')
    print("mAP is : {}".format(np.mean(AP_list)))


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap



def voc_eval(detpath, annopath, test_imgid_list, cls_name, ovthresh=0.5,
                 use_07_metric=False, use_diff=False):
  '''

  :param detpath:
  :param annopath:
  :param test_imgid_list: it 's a list that contains the img_name of test_imgs
  :param cls_name:
  :param ovthresh:
  :param use_07_metric:
  :param use_diff:
  :return:
  '''
  # 1. parse xml to get gtboxes

  # read list of images
  imagenames = test_imgid_list

  recs = {}
  for i, imagename in enumerate(imagenames):
    recs[imagename] = parse_rec(os.path.join(annopath, imagename+'.xml'))
    # if i % 100 == 0:
    #   print('Reading annotation for {:d}/{:d}'.format(
    #     i + 1, len(imagenames)))

  # 2. get gtboxes for this class.
  class_recs = {}
  num_pos = 0
  # if cls_name == 'person':
  #   print ("aaa")
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
    bbox = np.array([x['bbox'] for x in R])
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    num_pos = num_pos + sum(~difficult)  # ignored the diffcult boxes
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det} # det means that gtboxes has already been detected

  # 3. read the detection file
  detfile = os.path.join(detpath, "det_"+cls_name+".txt")
  with open(detfile, 'r') as f:
    lines = f.readlines()

  # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
  splitlines = [x.strip().split(' ') for x in lines]  # a list that include a list
  image_ids = [x[0] for x in splitlines]  # img_id is img_name
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids) # num of detections. That, a line is a det_box.
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]  #reorder the img_name

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]  # img_id is img_name
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # 4. get recall, precison and AP
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(num_pos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric=False)

  return rec, prec, ap




def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects



if __name__ == '__main__':
    eval_with_plac(np.inf,eval_dir='F:\Tree/test_images_simple\JPEGImages',
                   img_root='F:\Tree/test_images_simple\JPEGImages',showbox=True,
                   annotation_dir='F:\Tree/test_images_simple\Annotations')


    # eval_with_plac(np.inf,eval_dir='F:\SAR/test_images_complex_____\JPEGImages',
    #                img_root='F:\SAR/test_images_complex_____\JPEGImages',showbox=True,
    #                annotation_dir='F:\SAR/test_images_complex_____\Annotations')