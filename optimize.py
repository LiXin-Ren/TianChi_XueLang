# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 18:59:16 2018

@author: zxlation
@author: lhf
"""
import tensorflow as tf
import options

DECAY_STEPS = options.params.decay_steps
DECAY_RATE  = options.params.decay_rate
MAX_STEPS = options.params.max_steps

def learning_rate_exponential_decay(init_lr = 1e-4, global_step = None):
    """" Learning Decay
    
    Args:
        init_lr: initial learning rate.
        global_step: Global step to use for the decay computation. Must not be negative
    Returns:
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    """

    decayed_lr = tf.train.exponential_decay(learning_rate = init_lr,
                                            global_step = global_step,
                                            decay_steps = DECAY_STEPS,
                                            decay_rate = DECAY_RATE,
                                            staircase = True)
    
    return decayed_lr


def learning_rate_piecewise_decay(init_lr = 1e-4, global_step = None):
    """" Learning Decay
    
    Args:
        init_lr: initial learning rate.
        global_step: Global step to use for the decay computation. Must not be negative
    
    Returns:
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    """
    boundaries = [200000, 400000, 600000, 800000]
    values = [1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6]
    decayed_lr = tf.train.piecewise_constant(global_step, boundaries, values)
    
    return decayed_lr


def learning_rate_cosine_decay(init_lr = 1e-4, global_step = None):
    """" Learning Decay with cosine decay. It computes:
            global_step = min(global_step, decay_steps)
            cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            decayed_learning_rate = learning_rate * decayed
    Args:
        init_lr: initial learning rate.
        global_step: Global step to use for the decay computation. Must not be negative
    
    Returns:
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    """
    decay_steps = MAX_STEPS
    decayed_lr = tf.train.cosine_decay(init_lr, global_step, decay_steps)
    
    return decayed_lr


def learning_rate_inverse_time_decay(init_lr = 1e-4, global_step = None):
    """" Learning Decay. It is computed as:
        decayed_learning_rate = learning_rate/(1+decay_rate*global_step/decay_step)
    or if staircase = True:
        decayed_learning_rate = learning_rate/(1+decay_rate*floor(global_step/decay_step))
    
    Args:
        init_lr: initial learning rate.
        global_step: Global step to use for the decay computation. Must not be negative
    
    Returns:
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    """
    decay_steps = 50000
    decay_rate  = 0.5
    decayed_lr = tf.train.inverse_time_decay(learning_rate = init_lr,
                                             global_step = global_step,
                                             decay_steps = decay_steps,
                                             decay_rate  = decay_rate,
                                             staircase = True)
    
    return decayed_lr


def learning_rate_natural_exp_decay(init_lr = 1e-4, global_step = None):
    """" Learning Decay with natural exponential decay.It is computed as:
        decayed_learning_rate = learning_rate*exp(-decay_rate*global_step)
    
    Args:
        init_lr: initial learning rate.
        global_step: Global step to use for the decay computation. Must not be negative
    
    Returns:
        A scalar Tensor of the same type as learning_rate. The decayed learning rate.
    """
    decay_steps = 50000
    decay_rate  = 0.5
    decayed_lr = tf.train.exponential_time_decay(learning_rate = init_lr,
                                                 global_step = global_step,
                                                 decay_steps = decay_steps,
                                                 decay_rate  = decay_rate,
                                                 staircase = True)
    
    return decayed_lr


def optimize(loss, init_lr, global_step, method = 'AdamOptimizer'):
    """Sets up the training Ops.
    
    Args:
        loss: Loss tensor, from loss().
        init_lr: The initial learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    
    assert global_step != None, "global_step can not be None!"
    decayed_lr = learning_rate_piecewise_decay(init_lr, global_step)
    
    # Add summaries to learning rate.
    with tf.name_scope('learning_rate'):
        tf.summary.scalar('learning_rate', decayed_lr)
        tf.summary.histogram('histogram', decayed_lr)
    
    with tf.name_scope('optimization'):
        # NOTE the use of Optimizer: memory ×4
        if method is 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(learning_rate = decayed_lr)
        elif method is 'AdagradDAOptimizer':
            optimizer = tf.train.AdagradDAOptimizer(learning_rate = decayed_lr)
        elif method is 'AdagradOptimizer':
            optimizer = tf.train.AdagradOptimizer(earning_rate = decayed_lr)
        elif method is 'GradientDescentOptimizer':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = decayed_lr)
        elif method is 'ProximalAdagradOptimizer':
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate = decayed_lr)
        elif method is 'ProximalGradientDescentOptimizer':
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate = decayed_lr)
        elif method is 'RMSPropOptimizer':
            optimizer = tf.train.RMSPropOptimizer(learning_rate = decayed_lr)
        elif method is 'AdadeltaOptimizer':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate = decayed_lr)
        else:
            print("Invalid optimization method!")
            return
        
        # with a global_step to track the global step.
        train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op



