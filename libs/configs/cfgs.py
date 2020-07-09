import numpy as np
# dataset

DATASET_DIR = 'F:\Tree\dataset/sarship_train_single_channel.tfrecord'
SHORT_SIDE_LEN = 600
LONG_SIDE_LEN = 800
DATASET_NAME = 'sarship'
PIXEL_MEAN = [103.939, 116.779, 123.68]



# train parameters
BATCH_SIZE = 3
NET_NAME = 'resnet_v1_50'
NUM_CLASS = 1 + 1
WEIGHT_DECAY = {'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001}
PRETRAINED_WEIGHTS_DIR = "F:\Tree\FPN_TensorFlow-master\FPN_TensorFlow\data\pretrained_weights/resnet_v1_50.ckpt"
TRAINED_MODEL_DIR = 'F:\Tree\FPN_TensorFlow-master\FPN_TensorFlow/trained_model'
LR = 0.0001
MOMENTUM = 0.9
SUMMARY_PATH = 'F:\Tree\FPN_TensorFlow-master\FPN_TensorFlow\logs'
MAX_ITERATION = 50000



# anchor parameters
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]   # 需要改
ANCHOR_SCALES = [0.5,1.,2.]
ANCHOR_RATIOS = [0.5,0.75, 1, 2]
STRIDE = [4, 8, 16, 32, 64]



# rpn
LEVEL = ['P2', 'P3', 'P4', 'P5', "P6"]
SHARE_HEAD = True
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
RPN_MINIBATCH_SIZE = 256   # 选正负训练样本的个数 per image
RPN_POSITIVE_RATE = 0.5
RPN_TOP_K_NMS = 12000
MAX_PROPOSAL_NUM_TRAINING = 2000
MAX_PROPOSAL_NUM_INFERENCE = 1000
RPN_BBOX_STD_DEV = [0.1, 0.1, 0.25, 0.27]
BBOX_STD_DEV = [0.13, 0.13, 0.27, 0.26]





# Fast_RCNN
###################################
ROI_SIZE = 7
# the iou between different detections is
# less than HEAD_NMS_IOU_THRESHOLD
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3
FINAL_SCORE_THRESHOLD = 0.5
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_LOW_NEG_THRESHOLD = 0.1
FAST_RCNN_MINIBATCH_SIZE = 200
FAST_RCNN_POSITIVE_RATE = 0.33   # 比例是不是太低了  正例样本只有60个
DETECTION_MAX_INSTANCES = 200



DEBUG = False

LABEL_DICT = {'back_ground':0,
              'sarship':1}
LABEL_TO_NAME = {k:v for v,k in LABEL_DICT.items()}
print(LABEL_TO_NAME)




EVALUATE_DIR = 'F:\Tree\FPN_TensorFlow-master\FPN_TensorFlow' + '/output/evaluate_result_pickle/'
chekpoint_path = 'F:\Tree\FPN_TensorFlow-master\FPN_TensorFlow/trained_model\sarshipresnet_v1_50_55000model.ckpt'
GT_SAVE_PATH = 'F:\Tree\FPN_TensorFlow-master\FPN_TensorFlow\gt_boxes'

# train
CLIP_GRADIENT_NORM = 5.0


