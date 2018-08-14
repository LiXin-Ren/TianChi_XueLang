# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 18:51:35 2018

@author: zxlation
"""
import tensorflow as tf
import options

TOTAL_LOSS_COLLECTION = options.TOTAL_LOSS_COLLECTION

def loss_l2(real_HR, pred_HR):
    """Calculates the L2 loss from the real HR images and the predicted HR images.
            total_loss = ||real_HR - pred_HR||_L2 + η·WD
                       = ||real_HR - pred_HR||_2^2 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR

    Returns:
        total_loss: MSE between real HR images and predicted HR images and weight decay(optional).
    """
    with tf.name_scope('l2_loss'):
        mse_loss = tf.losses.mean_squared_error(real_HR, pred_HR)
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, mse_loss)
        
        # Attach a scalar summary to mse_loss
        tf.summary.scalar(mse_loss.op.name, mse_loss)
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION), name = 'total_loss_l2')


def loss_l1(real_HR, pred_HR):
    """Calculates the L1 loss from the real HR images and the predicted HR images.
              total_loss = ||real_HR - pred_HR||_L1 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR
    Returns:
        total loss: L1 loss between real clear images and predicted HR images.
    """
    with tf.name_scope('l1_loss'):
        abs_loss = tf.reduce_mean(tf.abs(real_HR - pred_HR))
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, abs_loss)
        tf.summary.scalar(abs_loss.op.name, abs_loss)
        
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))



def loss_lp(real_HR, pred_HR, p):
    """Calculates the Lp loss from the real HR images and the predicted HR images.
              total_loss = ||real_HR - pred_HR||_Lp + η·WD
       where 'WD' indicates weight decay. p should be smaller than 1.0
       --> try: (2/3)*[x^(3/2)]
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR
        p: the factor of the loss
    Returns:
        total loss: Lp loss between real clear images and predicted HR images.
    """
    
    alpha = 1e-2
    with tf.name_scope('lp_loss'):
        lp_loss = tf.reduce_mean(tf.pow(tf.abs(real_HR - pred_HR) + alpha, p))
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, lp_loss)
        tf.summary.scalar(lp_loss.op.name, lp_loss)
        
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))


def loss_cross(real_HR, pred_HR): 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred_HR, labels=real_HR, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
   
    return cross_entropy_mean
