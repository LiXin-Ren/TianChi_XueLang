# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:16:14 2018

@author: LHF
"""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import options

XAVIER_INIT = options.params.xavier_init

USE_FP16 = options.params.use_fp16

TOTAL_LOSS_COLLECTION = options.TOTAL_LOSS_COLLECTION

WEIGHT_DECAY = 0.0

def variable_summaries(var, name):
    """Add a lot of summaries to a tensor.
    Args:
        var: a tensor as variables/activations
        name: scope name
    Returns:
        None
    """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def variable_initializer(prev_units, curr_units, kernel_size, stddev_factor = 1.0):
    """Initialization for CONV2D in the style of Xavier Glorot et al.(2010).
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs.
    ArgS:
        prev_units: The number of channels in the previous layer.
        curr_units: The number of channels in the current layer.
        stddev_factor: 
    Returns:
        Initial value of the weights of the current conv/transpose conv layer.
    """
    
    if XAVIER_INIT:
        stddev = np.sqrt(stddev_factor/(np.sqrt(prev_units*curr_units)*kernel_size*kernel_size))
    else:
        stddev = 0.01
    
    return tf.truncated_normal_initializer(mean = 0.0, stddev = stddev)


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if USE_FP16 else tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
    return var


def variable_with_weight_decay(name, shape, weight_decay = None, is_conv = True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        is_conv: Initialization mode for variables: 'conv' and 'convT'

    Returns:
        Variable Tensor
    """
    if is_conv == True:
        initializer = variable_initializer(shape[2], shape[3], shape[0], stddev_factor = 1.0)
    else:
        initializer = variable_initializer(shape[3], shape[2], shape[0], stddev_factor = 1.0)
    
    var = variable_on_cpu(name, shape, initializer)
    if weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name = 'weight_loss')
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, weight_decay)
    
    return var


def pooling(inputs, kernel_size, stride, mode = 'max', name = None):
    """Excute max or average pooling on inputs.
    
    Args:
        inputs: input tensor
        kernel_size: kernel size for pooling
        stride: stride used in pooling
        mode: 'max' | 'avg'
        name: name for this operation
    Returns:
        Tensor that is pooled, with size changed.
    """
    
    strides = [1, stride, stride, 1]
    ksize = [1, kernel_size, kernel_size, 1]
    if mode == 'max':
        inputs = tf.nn.max_pool(inputs, ksize, strides, padding = 'SAME', name = name)
    elif mode == 'avg':
        inputs = tf.nn.avg_pool(inputs, ksize, strides, padding = 'SAME', name = name)
    else:
        raise ValueError("Unknown pooling %s!" % mode)
        
    return inputs


def conv2d_layer(inputs, kernel_shape, stride = 1, weight_decay = 0.0, name = None):
    """A common convolutional layer that excutes 'SAME' convolution and bias addition.
    
    Args:
        inputs: 4D input tensor.
        kernel_shape: shape of convolution kernels, [batch_size, batch_size, in_maps, out_maps].
        weight_decay: weight decay factor.
        name: name for this operation.
    Returns:
        result tensor of conv & add operations.
    """
    with tf.variable_scope(name):
        W = variable_with_weight_decay('weights', kernel_shape, weight_decay, is_conv = True)
        b = variable_on_cpu('biases', [kernel_shape[3]], tf.constant_initializer())
        variable_summaries(W, name + 'weights')
        variable_summaries(b, name + 'biases')
        
        strides = [1, stride, stride, 1]
        padding = 'SAME'
        conv_name = name + 'conv_op'
        inputs = tf.nn.conv2d(inputs, W, strides, padding, name = conv_name)
    
    add_name = name + 'add_op'
    inputs = tf.nn.bias_add(inputs, b, name = add_name)
    
    return tf.nn.relu(inputs, name + 'relu')


def conv2d_layer_no_act(inputs, kernel_shape, stride = 1, weight_decay = 0.0, name = None):
    """A common convolutional layer that excutes 'SAME' convolution and bias addition.
    
    Args:
        inputs: 4D input tensor.
        kernel_shape: shape of convolution kernels, [batch_size, batch_size, in_maps, out_maps].
        weight_decay: weight decay factor.
        name: name for this operation.
    Returns:
        result tensor of conv & add operations.
    """
    with tf.variable_scope(name):
        W = variable_with_weight_decay('weights', kernel_shape, weight_decay, is_conv = True)
        b = variable_on_cpu('biases', [kernel_shape[3]], tf.constant_initializer())
        variable_summaries(W, name + 'weights')
        variable_summaries(b, name + 'biases')
        
        strides = [1, stride, stride, 1]
        padding = 'SAME'
        conv_name = name + 'conv_op'
        inputs = tf.nn.conv2d(inputs, W, strides, padding, name = conv_name)
    
    add_name = name + 'add_op'
    inputs = tf.nn.bias_add(inputs, b, name = add_name)
    
    return inputs


def adaptive_block(inputs, nlayer, wdecay = 0.0, name = None):
    maps = int(inputs.get_shape()[3])
    ksize_m = 2
    stride_m = 2
    
    ksize = 3
    stride = 1
    pool_mode = 'max'
    
    Kshape = [ksize, ksize, maps, 2*maps]
    inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + 'conv2d_layer_upsample'))
    
    for i in range(nlayer):
        Kshape = [ksize, ksize, 2*maps, 2*maps]
        inputs = conv2d_layer(inputs, Kshape, stride,  wdecay, (name + ('conv2d_layer_%d' % i)))
    inputs = pooling(inputs, ksize_m, stride_m, pool_mode, 'maxpool')
    return inputs


def fcn(inputs, maps_s1, maps_s2, maps_s3):
    std = 0.01
    mean = 0.0
    dtype = tf.float32
    c, w, d = inputs.get_shape().as_list()[1:4]
    length = c * w * d
    inputs = tf.reshape(inputs, [-1, length], name="reshape")
    with tf.variable_scope("fc1"):
        fc1_weight = tf.Variable(tf.truncated_normal([length, maps_s1], mean, std, dtype, name="fc1_Weight"))
        fc1_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s1], name="fc1_bias"))
        fc1 = tf.matmul(inputs, fc1_weight)
        fc1 = tf.nn.bias_add(fc1, fc1_bias)
        fc1 = tf.nn.relu(fc1)
    fc1_drop = tf.nn.dropout(fc1, 0.5, name = "fc1_drop")  
    
    with tf.variable_scope("fc2"):
        fc2_weight = tf.Variable(tf.truncated_normal([maps_s1, maps_s2], mean, std, dtype, name="fc2_Weight"))
        fc2_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s2], name="fc2_bias"))
        fc2 = tf.matmul(fc1_drop, fc2_weight)
        fc2 = tf.nn.bias_add(fc2, fc2_bias)
        fc2 = tf.nn.relu(fc2)
    fc2_drop = tf.nn.dropout(fc2, 0.5, name = "fc2_drop")
    
    with tf.variable_scope("fc3"):
        fc3_weight = tf.Variable(tf.truncated_normal([maps_s2, maps_s3], mean, std, dtype, name = "fc3_Weight"))
        fc3_bias = tf.Variable(tf.constant(0.0, dtype, shape=[maps_s3], name = "fc3_bias"))
        fc3 = tf.matmul(fc2_drop, fc3_weight)
        fc3 = tf.nn.bias_add(fc3, fc3_bias)
        fc3 = tf.nn.relu(fc3)
    return fc3


def adaptive_classification_model(inputs):
    inp_maps = int(inputs.get_shape()[3])
    num_block = 4
    init_maps = 32
    init_ksize = 3
    nlayer = 2
    stride = 1
    wdecay = 0.0
    Kshape = [init_ksize, init_ksize, inp_maps, init_maps]
    inputs = conv2d_layer_no_act(inputs, Kshape, stride,  wdecay, 'conv2d_layer_init')
    
    for i in range(num_block):
        inputs = adaptive_block(inputs, nlayer, wdecay = 0.0, name = ('resbk%d/' % i))
        
    inputs = fcn(inputs, 1024, 256, 1)
    return inputs
