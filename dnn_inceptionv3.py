import numpy as np
import tensorflow as tf
from inception_v3 import inception_v3
#from read_record import read_and_decode         # tfrecords文件
from inception_utils import inception_arg_scope
import os
import input
import tensorlayer as tl
from scipy import misc
import csv
from sklearn.model_selection import train_test_split

slim = tf.contrib.slim
LogDir = "train_logs"

testDir = "data/test/"
n_epoch = 1000
learning_rate = 0.0001
print_freq = 2
batch_size = 32


x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
y_ = tf.placeholder(tf.int32, shape=[None, 1], name='y_')

net_in = tl.layers.InputLayer(x, name='input_layer')                #input

with slim.arg_scope(inception_arg_scope()):
    network = tl.layers.SlimNetsLayer(
        prev_layer=net_in, slim_layer=inception_v3,
        slim_args={'num_classes': 256, 'is_training': True},
        name='')

network = tl.layers.DenseLayer(network, n_units=1, act=tf.sigmoid, name='dense2')
y_op = network.outputs                                              #ouput

print("y_op.shape: ", y_op.shape)
print("y_.shape: ", y_.shape)

cost = tf.losses.mean_squared_error(labels=y_, predictions=y_op)
loss = tf.reduce_mean(tf.square(tf.cast(y_, tf.float32) - y_op))
train_params = network.all_params[384:]                             #  训练的参数

#print("训练参数个数: ", len(train_params))

train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, var_list = train_params)


def train(X_train, Y_train, X_val, Y_val):
    with tf.Session() as sess:
        #global_step = tf.train.get_or_create_global_step()
        saver = tf.train.Saver(tf.global_variables())
        summ_op = tf.summary.merge_all()

        print("initialize")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        params = tl.files.load_npz('', 'models/model_inceptionV3.npz')
        params = params[0:384]
        #print("当前参数大小: ", len(params))
        tl.files.assign_params(sess, params=params, network=network)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        step = 0

        print("Defining Summary Writer...")
        summary_writer = tf.summary.FileWriter(LogDir, sess.graph)
        min_train_loss = 1
        min_valid_loss = 1
        try:
            while not coord.should_stop():
                batch_x, batch_y = sess.run([X_train, Y_train])
                _, losses, pre_y = sess.run([train_op, loss, y_op], feed_dict = {x: batch_x, y_: batch_y})

                if step % 100 == 0:
                    #统计train 和val的loss
                    if losses < min_train_loss:
                        min_train_loss = losses

                    #Valid
                    preList = []
                    for name in X_val:
                        val_image = misc.imread(name)
                        val_image = misc.imresize(val_image, (299, 299))
                        val_image = val_image[np.newaxis, :]

                        pre_y = sess.run([y_op], feed_dict={x: val_image})
                        preList.append(pre_y)

                    val_loss = np.mean(np.square(np.array(preList) - np.array(Y_val)))
                    if val_loss < min_valid_loss:
                        min_valid_loss = val_loss
                    with open("Records/train_records.txt", "a") as file:
                        format_str = "%d\t%.6f\t%.6f\t\t\t%.6f\t%.6f\n"
                        file.write(str(format_str) %(step, losses, min_train_loss, val_loss, min_valid_loss))

                if step % 200 == 0:
                    #写checkpoint文件，计算测试集
                    summary_str = sess.run(summ_op)
                    summary_writer.add_summary(summary_str, step)
                    chckpoint_path = os.path.join(LogDir, 'model.ckpt')
                    print("saving checkpoint into %s-%s" %(chckpoint_path, step))
                    saver.save(sess, chckpoint_path, global_step=step)

                    print("test....")
                    testList = os.listdir(testDir)

                    with open("Records/test_"+str(step) + ".csv", "w", newline="") as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['filename', 'probability'])
                        for name in testList:
                            test_image = misc.imread(testDir+name)
                            test_image = misc.imresize(test_image, (299, 299))

                            test_image = test_image[np.newaxis, :]       #test_image.reshape((1, 299, 299, 3))
    #                        print("test_imageNew.shape: ", test_image.shape)

                            pre_y = sess.run([y_op], feed_dict={x: test_image})

                            csv_writer.writerow([name, round(float(pre_y[0][0]), 4)])


                step = step + 1
        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached")
        finally:
            coord.request_stop()

    coord.join(threads)

    print("ok")

if __name__ == "__main__":
    images, labels = input.GetFileNameList()
    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.01)

    X_train, Y_train = input.GetBatchFromFile_Train(X_train, Y_train, batch_size)

    #X_val, Y_val = input.GetBatchFromFile_Valid(X_val, Y_val, batch_size)

    # print("X_train.shape: ", X_train.shape)
    # print("y_train.shape: ", y_train.shape)

    X_train = tf.image.resize_images(X_train, [299, 299])
    # print("X_train.shape: ", X_train.shape)

    print("y_.shape: ", y_.shape)

    print("y_train.shape: ", Y_train.shape)

    train(X_train, Y_train, X_val, Y_val)

#
# def main(argv=None):  # pylint: disable = unused - argument
#     if not TrainFromExist:
#         if tf.gfile.Exists(LogDir):
#             tf.gfile.DeleteRecursively(LogDir)
#         tf.gfile.MakeDirs(LogDir)
#     else:
#         if not tf.gfile.Exists(ExistModelDir):
#             raise ValueError("Train from existed model, but the target dir does not exist.")
#
#         if not tf.gfile.Exists(LogDir):
#             tf.gfile.MakeDirs(LogDir)
#     train()
#
#
# if __name__ == '__main__':
#     tf.app.run()