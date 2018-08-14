import os
import time
from recordutil import *
import numpy as np
# from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152
# from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from tensorlayer.layers import *
# from scipy.misc import imread, imresize
# from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from inception_resnet_v2 import (inception_resnet_v2_arg_scope, inception_resnet_v2)
from scipy.misc import imread, imresize
from tensorflow.python.ops import variables
import tensorlayer as tl

slim = tf.contrib.slim
try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception(
    "{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))

n_epoch = 200
learning_rate = 0.0001
print_freq = 2
batch_size = 32
## InceptionV3 / All TF-Slim nets can be merged into TensorLayer
x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')    # 输出

net_in = tl.layers.InputLayer(x, name='input_layer')
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    network = tl.layers.SlimNetsLayer(
        prev_layer=net_in,
    slim_layer=inception_resnet_v2,
    slim_args={
        'num_classes': 1001,
        'is_training': True,
        },
    name='InceptionResnetV2')    # <-- the name should be the same with the ckpt model

# network = fc_layers(net_cnn)
sess = tf.InteractiveSession()
network.print_params(False)
# network.print_layers()
saver = tf.train.Saver()

# 加载预训练的参数
# tl.files.assign_params(sess, npz, network)

tl.layers.initialize_global_variables(sess)

saver.restore(sess, "inception_resnet_v2.ckpt")
print("Model Restored")
all_params = sess.run(network.all_params)
np.savez('inception_resnet_v2.npz', params=all_params)
sess.close()