# -*- coding utf-8 -*-
import tensorflow as tf
import sys
from scipy import misc
import os
import random

IMG_HEIGHT = 192
IMG_WIDTH = 256
IMG_CHANNELS = 3

flaw_dir = "data/flaw/"
norm_dir = "data/norm/"
data_dir = "data/"


def GetFileNameList():

    name_flaw = os.listdir(flaw_dir)
    name_flaw = [data_dir + "flaw/" + name for name in name_flaw]
    label_flaw = [1 for i in range(len(name_flaw))]

    name_norm = os.listdir(norm_dir)
    name_norm = [data_dir + "norm/" + name for name in name_norm]
    label_norm = [0 for i in range(len(name_norm))]

    input_names = name_flaw + name_norm
    labels = label_flaw + label_norm

    num_examples = len(input_names)
    if num_examples != len(labels):
        raise ValueError("Number of LR images %s does not match number of HR images %s!" %
                         (num_examples, len(labels)))

    cc = list(zip(input_names, labels))
    random.shuffle(cc)
    input_names[:], labels[:] = zip(*cc)

    return input_names, labels


def GetBatchFromFile_Train(input_names, labels, BatchSize):
    '''
    Args:
        rawDir: Directory of raw and segmante images
        BatchSize: batch size
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
        label_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
    '''

    imageList = tf.cast(input_names, tf.string, name='RawImage')
    labelList = tf.cast(labels, tf.int32, name='Label')

    # Make an input queue
    InputQueue = tf.train.slice_input_producer([imageList, labelList],
                                               num_epochs=200,
                                               shuffle=False,
                                               capacity=200,
                                               shared_name=None,
                                               name='SliceInputProducer')

    # Read one example from input queue
    imageContent = tf.read_file(InputQueue[0], name='ReadrawImage')
    labelContent = InputQueue[1]
    # labelContent =

    # Decode the jpeg image format
    rawImage = tf.image.decode_jpeg(imageContent, channels=3, name='DecodeRawImage')


    with tf.name_scope('SetShape'):
        rawImage.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    #    labelContent.set_shape([1])
    imageBatch, labelBatch = tf.train.shuffle_batch([rawImage, labelContent],
                                                          batch_size=BatchSize,
                                                          num_threads=8,
                                                          capacity=10 * BatchSize,
                                                          min_after_dequeue=BatchSize * 2,
                                                          name='SuffleBatch')

    #tf.summary.image('train_GIBBS_images', rawImageBatch, max_outputs=4)
    #tf.summary.image('train_CLEAR_images', segImageBatch, max_outputs=4)
    # Cast to tf.float32
    with tf.name_scope('CastToFloat'):
        labelBatch = tf.reshape(labelBatch, [BatchSize, 1])
        imageBatch = tf.cast(imageBatch, tf.float32)
       # labelBatch = tf.cast(labelBatch, tf.int8)

    # Normalization
    with tf.name_scope('Normalization'):
        imageBatch = imageBatch / 255.0

    return imageBatch, labelBatch


