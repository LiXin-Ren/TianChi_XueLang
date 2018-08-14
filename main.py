# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:17:57 2018

@author: lhf
"""
from datetime import datetime
from scipy import misc
import tensorflow as tf
import numpy as np
import optimize
import options
#import termcolor
import warnings
import model
import loss
import random
import time
import os

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_img_dir = options.params.train_image_dir
valid_img_dir = options.params.valid_image_dir

EPOCH = 100000

img_H = options.ImageH
img_W = options.LabelW

norm = 255.0
nChann = options.CHANNELS

train_log_dir = options.params.train_log_dir
learning_rate = options.params.learning_rate
Batch_size = options.params.batch_size
max_steps = options.params.max_steps

train_from_exist = options.params.train_from_exist
exist_model_dir = options.params.exist_model_dir


def Cal_Accuracy(model_batch, lab_batch):
    if model_batch.shape != lab_batch.shape:
        errstr = "image does not match label!"
        raise ValueError(errstr)
    correct_pred = np.abs(model_batch - lab_batch)
#    correct_pred = np.equal(np.argmax(model_batch, 1), np.argmax(lab_batch, 1))
    accuracy = np.mean(correct_pred, dtype = np.float32)
    return accuracy

'''
def get_valid_batch(valid_image_dir):
    valid_names_list = os.listdir(valid_image_dir)
    num_name = len(valid_names_list)
    valid_img_batch = np.zeros([num_name, img_H, img_W, nChann], dtype = np.float32)
    valid_lab_batch = np.zeros([num_name, 1], dtype = np.float32)
    
    for i in range(num_name):
        path_img = os.path.join(valid_image_dir, valid_names_list[i])
        lab = int(valid_names_list[i].split('_')[4])
        img = (misc.imread(path_img) / norm).astype(np.float32)

        valid_img_batch[i, :, :, :] = img
        valid_lab_batch[i, :] = lab
    return valid_img_batch, valid_lab_batch
'''

def get_train_batch(batch_names_list, batch_size):
    '''
    Get image batch and label batch
    '''
    
    train_img_batch = np.zeros([batch_size, img_H, img_W, nChann], dtype = np.float32)
    train_lab_batch = np.zeros([batch_size, 1], dtype = np.float32)
    for i in range(batch_size):
        path_img = os.path.join(train_img_dir, batch_names_list[i])
        lab = int(batch_names_list[i].split('_')[4])
            
        img = (misc.imread(path_img) / norm).astype(np.float32)   
        #lab_batch = (lab / norm).astype(np.float32)
        train_img_batch[i, :, :, :] = img
        train_lab_batch[i, :] = lab
    return train_img_batch, train_lab_batch
 

def restore_model(sess, saver, exist_model_dir, global_step):
    log_info = "Restoring Model From %s..." % exist_model_dir
    print(log_info)
    ckpt = tf.train.get_checkpoint_state(exist_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, init_step))
    else:
        print('No Checkpoint File Found!')
        return
    
    return init_step       
        
    
def train():
    global_step = tf.train.get_or_create_global_step()
    
    Img_Batch = tf.placeholder(dtype = tf.float32, shape = [None, img_H, img_W, nChann])
    Lab_Batch = tf.placeholder(dtype = tf.float32, shape = [None, 1])
    
    print("Building Computation Graph...")
    pred_model = model.adaptive_classification_model(Img_Batch)
    
    train_loss = loss.loss_l2(Lab_Batch, pred_model)
    
    train_op = optimize.optimize(train_loss, learning_rate, global_step)
    
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 3)
    
    summ_op = tf.summary.merge_all()
    
    config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
    sess = tf.Session(config = config)
    
    print("Defining Summary Writer...")
    summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    
    step = 0
    if train_from_exist:
        step = restore_model(sess, saver, exist_model_dir, global_step)
    else:
        print("Initializing Variables...")
        sess.run(tf.global_variables_initializer())
    
    min_loss = float('Inf')
    max_acury = float('-Inf')
    
    train_names_list = os.listdir(train_img_dir)
    num_names = len(train_names_list)
    index = np.arange(int(num_names / Batch_size))
    index = index * Batch_size
    print("Starting To Train...")
    
    for i in range(EPOCH):
        random.shuffle(train_names_list)
        for k in index:
            
            step += 1
            start_time = time.time()
        
            Batch_names_list = train_names_list[k:k + Batch_size]
            batch_train_img, batch_train_lab = get_train_batch(Batch_names_list, Batch_size)
            
            feed_dict = {Img_Batch: batch_train_img, Lab_Batch: batch_train_lab}
            _, model_loss, pred_batch = sess.run([train_op, train_loss, pred_model], feed_dict = feed_dict)
            accuracy = Cal_Accuracy(pred_batch, batch_train_lab)
            
            duration = time.time() - start_time
            if (step + 1) % 100 == 0:
                examples_per_second = Batch_size/duration
                seconds_per_batch = float(duration)
                if model_loss < min_loss: min_loss = model_loss
                if accuracy > max_acury: max_acury = accuracy
                with open("records/train_records.txt", "a") as files:
                    format_str = "%d\t%.6f\t%.6f\n"
                    files.write(str(format_str)%(step + 1, model_loss, accuracy)) 
           
                
                print('%s ---- step #%d' % (datetime.now(), step + 1))
                print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
                print('  accuracy = %.6f\t MAX_ACURY = %.6f' % (accuracy, max_acury))
                print('  ' + '%.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch))

            if ((step + 1) % 200 == 0) or ((step + 1) == max_steps):
                summary_str = sess.run(summ_op, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step + 1)
                    
            if (step == 0) or ((step + 100) % 500 == 0) or ((step + 1) == max_steps):
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                print("saving checkpoint into %s-%d" % (checkpoint_path, step + 1))
                saver.save(sess, checkpoint_path, global_step = step + 1)
                    
    summary_writer.close()
    sess.close()
    
    
def main(argv = None):  
    if not train_from_exist:
        if tf.gfile.Exists(train_log_dir):
            tf.gfile.DeleteRecursively(train_log_dir)
        tf.gfile.MakeDirs(train_log_dir)
    else:
        if not tf.gfile.Exists(exist_model_dir):
            raise ValueError("Train from existed model, but the target dir does not exist.")
        
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
    
