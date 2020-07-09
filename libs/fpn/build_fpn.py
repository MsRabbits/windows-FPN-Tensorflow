import tensorflow.contrib.slim as slim
import tensorflow as tf

from libs.configs import cfgs




def get_feature_maps(share_net):
    '''
        Compared to https://github.com/KaimingHe/deep-residual-networks, the implementation of resnet_50 in slim
        subsample the output activations in the last residual unit of each block,
        instead of subsampling the input activations in the first residual unit of each block.
        The two implementations give identical results but the implementation of slim is more memory efficient.

        SO, when we build feature_pyramid, we should modify the value of 'C_*' to get correct spatial size feature maps.
        :return: feature maps
    '''

    with tf.variable_scope('get_feature_maps'):
        if cfgs.NET_NAME == 'resnet_v1_50':
            feature_maps_dict = {
                'C2': share_net['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # [56, 56]
                'C3': share_net['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # [28, 28]
                'C4': share_net['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # [14, 14]
                'C5': share_net['resnet_v1_50/block4']  # [7, 7]
            }
        elif cfgs.NET_NAME == 'resnet_v1_101':
            feature_maps_dict = {
                'C2': share_net['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # [56, 56]
                'C3': share_net['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # [28, 28]
                'C4': share_net['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # [14, 14]
                'C5': share_net['resnet_v1_101/block4']  # [7, 7]
            }
        else:
            raise Exception('get no feature maps')

        return feature_maps_dict


def build_feature_pyramid(share_net):
    '''
    reference: https://github.com/CharlesShang/FastMaskRCNN
    build P2, P3, P4, P5
    :return: multi-scale feature map
    '''

    feature_maps_dict = get_feature_maps(share_net)
    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY[cfgs.NET_NAME])):
            feature_pyramid['P5'] = slim.conv2d(feature_maps_dict['C5'],
                                                num_outputs=256,
                                                kernel_size=[1, 1],
                                                stride=1,
                                                scope='build_P5')

            feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                    kernel_size=[2, 2], stride=2, scope='build_P6')
            # P6 is down sample of P5

            for layer in range(4, 1, -1):
                p, c = feature_pyramid['P' + str(layer + 1)], feature_maps_dict['C' + str(layer)]
                up_sample_shape = tf.shape(c)
                up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                             name='build_P%d/up_sample_nearest_neighbor' % layer)

                c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                scope='build_P%d/reduce_dimension' % layer)
                p = up_sample + c
                p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                feature_pyramid['P' + str(layer)] = p

    return feature_pyramid