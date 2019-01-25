import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
def attention_module(upconv, skipconv):
    """
    Implementation of Attention Module (2 pool in soft mask branch)
    Input:
    --- upconv: Module input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- skipconv: Module input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- name: Module name
    Output:
    --- outputs: Module output
    """
    # Sigmoid
    masks_4 = tf.nn.sigmoid(upconv, "mask_sigmoid")
    # Fusing
    with tf.name_scope("fusing"), tf.variable_scope("fusing"):
        outputs = tf.multiply(skipconv, masks_4, name="fuse_mul")
        outputs = tf.add(upconv, outputs, name="fuse_add")
        return outputs
def weight_variable(shape, name):
    nl = shape[0]*shape[1]*shape[3]
    std = 2/nl
    std = np.sqrt(std)
    initial = tf.truncated_normal(shape, mean=0, stddev = 0.001)
    weight = tf.Variable(initial_value=initial, name=name)
    #weight = tf.Variable(tf.random_normal(shape, stddev=0.001), dtype=tf.float32)
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(0.001)(weight))
    return weight

def weight_variable_devonc(shape, stddev=0.001):
    nl = shape[0] * shape[1] * shape[2]
    std = 2 / nl
    std = np.sqrt(std)
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev = 0.001))

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)

def conv2d(x, w, s=1):
    return tf.nn.conv2d(x, w, strides=[1, s, s, 1],
                        padding='SAME')
def relu(x):
    return tf.nn.relu(x, name='relu')

def dilation_layer(x,input_channel, output_channel,
                   k_size=3, rate=2, padding='SAME'):
    w = weight_variable([k_size, k_size, input_channel,
                         output_channel], 'weight')
    b = bias_variable([output_channel], 'bias')
    dilation_result = tf.nn.atrous_conv2d(x,w,rate,padding)+b
    return dilation_result

def conv_layer(x,input_channel,output_channel,
               k_size=3,stride=1):
    w = weight_variable([k_size, k_size, input_channel,
                         output_channel], 'weight')
    b = bias_variable([output_channel], 'bias')
    conv_result = conv2d(x, w, stride)+b
    return conv_result
def deconv2d(x, stride):
    x_shape = x.get_shape().as_list()
    # x_shape = tf.shape(x)
    #output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    #x_shape[3] = x_shape[3]//2
    #W = tf.stack([2, 2, x_shape[3] // 2, x_shape[3]])
    #wshape = tf.stack([2, 2, x.get_shape()[3] // 2, x.get_shape()[3]])
    wshape = [2, 2, x_shape[3]//2, x_shape[3]]
    x_shape[1] = x_shape[1] * 2
    x_shape[2] = x_shape[2] * 2
    x_shape[3] = x_shape[3] // 2
    W = weight_variable_devonc(wshape, 0.001)
    return tf.nn.conv2d_transpose(x, W, x_shape, strides=[1, stride, stride, 1], padding='VALID')

def transpose_conv2d(x, stride, channel):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, channel])
    #W = tf.stack([2, 2, x_shape[3] // 2, x_shape[3]])
    wshape = tf.stack([2, 2, channel, x.get_shape()[3]])
    W = weight_variable_devonc(wshape, 0.001)
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

# only use this function

def lst_unet(input_1, input_2, input_channel = 1, is_bntraining = True):
    concat_1_2 = tf.concat([input_1, input_2], 3)
    add_1_2 = input_1+input_2
    #minus_1_2 = input_1 - input_2
    concat_1_2 = tf.concat([concat_1_2, add_1_2], 3)
    #concat_1_2 = tf.concat([concat_1_2, minus_1_2], 3)
    conv1_1 = conv_layer(concat_1_2, input_channel, 64, 3, 1)
    conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_bntraining)
    conv1_1 = relu(conv1_1)
    conv1_2 = conv_layer(conv1_1, 64, 64, 3, 1)
    conv1_2 = tf.layers.batch_normalization(conv1_2, training=is_bntraining)
    conv1_2 = relu(conv1_2)

    conv2_1 = tf.nn.avg_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv2_2 = conv_layer(conv2_1, 64, 128, 3, 1)
    conv2_2 = tf.layers.batch_normalization(conv2_2, training=is_bntraining)
    conv2_2 = relu(conv2_2)
    conv2_3 = conv_layer(conv2_2, 128, 128, 3, 1)
    conv2_3 = tf.layers.batch_normalization(conv2_3, training=is_bntraining)
    conv2_3 = relu(conv2_3)

    conv3_1 = tf.nn.avg_pool(conv2_3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv3_2 = conv_layer(conv3_1, 128, 256, 3, 1)
    conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_bntraining)
    conv3_2 = relu(conv3_2)
    conv3_3 = conv_layer(conv3_2, 256, 256, 3, 1)
    conv3_3 = tf.layers.batch_normalization(conv3_3, training=is_bntraining)
    conv3_3 = relu(conv3_3)

    conv4_1 = tf.nn.avg_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    conv4_2 = conv_layer(conv4_1, 256, 512, 3, 1)
    conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_bntraining)
    conv4_2 = relu(conv4_2)
    conv4_3 = conv_layer(conv4_2, 512, 512, 3, 1)
    conv4_3 = tf.layers.batch_normalization(conv4_3, training=is_bntraining)
    conv4_3 = relu(conv4_3)

    conv43 = deconv2d(conv4_3, 2)
    conv3_4 = tf.concat([conv43, conv3_3], 3)
    conv3_4 = attention_module(conv43,conv3_3)
    conv3_5 = conv_layer(conv3_4, 256, 256, 3, 1)
    conv3_5 = tf.layers.batch_normalization(conv3_5, training=is_bntraining)
    conv3_5 = relu(conv3_5)
    conv3_6 = conv_layer(conv3_5, 256, 256, 3, 1)
    conv3_6 = tf.layers.batch_normalization(conv3_6, training=is_bntraining)
    conv3_6 = relu(conv3_6)

    conv32 = deconv2d(conv3_6, 2)
    conv2_4 = tf.concat([conv32, conv2_3], 3)
    conv2_4 = attention_module(conv32, conv2_3)
    conv2_5 = conv_layer(conv2_4, 128, 128, 3, 1)
    conv2_5 = tf.layers.batch_normalization(conv2_5, training=is_bntraining)
    conv2_5 = relu(conv2_5)
    conv2_6 = conv_layer(conv2_5, 128, 128, 3, 1)
    conv2_6 = tf.layers.batch_normalization(conv2_6, training=is_bntraining)
    conv2_6 = relu(conv2_6)

    conv21 = deconv2d(conv2_6, 2)
    conv1_3 = tf.concat([conv21, conv1_2], 3)
    conv1_3 = attention_module(conv21, conv1_2)
    conv1_4 = conv_layer(conv1_3, 64, 64, 3, 1)
    conv1_4 = tf.layers.batch_normalization(conv1_4, training=is_bntraining)
    conv1_4 =relu(conv1_4)
    conv1_5 = conv_layer(conv1_4, 64, 64, 3, 1)
    conv1_5 = tf.layers.batch_normalization(conv1_5, training=is_bntraining)
    conv1_5 = relu(conv1_5)
    conv1_6 = conv_layer(conv1_5, 64, 32, 3, 1)
    conv1_6 = tf.layers.batch_normalization(conv1_6, training=is_bntraining)
    conv1_6 = relu(conv1_6)
    conv1_7 = conv_layer(conv1_6, 32, 1, 3, 1)
    # conv1_7 = tf.layers.batch_normalization(conv1_7, training=is_bntraining)
    return conv1_7


