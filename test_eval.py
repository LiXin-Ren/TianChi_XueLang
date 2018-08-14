from __future__ import absolute_import

from datetime import datetime
import tensorflow as tf

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy import misc

HEIGHT = 256
WIDTH = 256

file_dir = "../preprocess/preprocessed_images/"

rawNameList = unet_input.getFileList(file_dir)
TOTAL_VALID_IMAGES = len(rawNameList)

x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
y_ = tf.placeholder(tf.int32, shape=[None, 1], name='y_')
net_in = tl.layers.InputLayer(x, name='input_layer')
# network = inception_v3(net_in, num_classes = 256)
# network = tl.layers.FlattenLayer(network)
with slim.arg_scope(inception_arg_scope()):
    network = tl.layers.SlimNetsLayer(
        prev_layer=net_in, slim_layer=inception_v3,
        slim_args={'num_classes': 256, 'is_training': True},
        name='')
print("v3 params: ", len(network.all_params))
# network = tl.layers.DenseLayer(network, n_units = 256, act = tf.nn.relu, name = 'dense1')
# network = tl.layers.DropoutLayer(network, keep = prob, name = 'drop1')
network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name='dense2')
print("dense params: ", len(network.all_params))
y = network.outputs

# y_op = tf.argmax(tf.nn.softmax(y), axis = 1)
y_op = tf.sigmoid(y)
print("y_op.shape: ", y_op.shape)
print("y_.shape: ", y_.shape)
# cost = tl.cost.cross_entropy(y, y_, name = 'cost')
# y_onehot = tf.one_hot(indices = tf.cast(y_, tf.int32), depth = 120)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits = y))
cost = tf.losses.mean_squared_error(labels=y_, predictions=y_op)
correct_prediction = tf.equal(tf.cast(y, tf.float32), tf.cast(y_, tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# acc = tf.metrics.accuracy(labels = y_, predictions = y_op)
train_params = network.all_params[384:]  # 训练的参数
print("训练参数个数: ", len(train_params))
# train_params.extend(dense1.all_params)
# train_params.extend(drop1.all_params)
# train_params.extend(dense2.all_params)
# print("训练参数个数: ", len(train_params))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=train_params)

X_train, y_train = input.GetBatchFromFile_Train(batch_size)
print("img.shape: ", X_train.shape)
print("label.shape: ", y_train.shape)

img = tf.image.resize_images(X_train, [299, 299])
# X_train, y_train = tf.train.shuffle_batch([img, label], batch_size = batch_size, capacity = 200, min_after_dequeue = 100)

with tf.Session() as sess:
    # sess = tf.InteractiveSession()
    # tl.layers.initialize_global_variables(sess)
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(tf.global_variables())
    summ_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("initialize")

    params = tl.files.load_npz('', 'models/model_inceptionV3.npz')
    params = params[0:384]
    print("当前参数大小: ", len(params))
    tl.files.assign_params(sess, params=params, network=network)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    epoch_i = 0

    #  step = 0
    # config = tf.ConfigProto()
    # config.log_device_placement = LogDevicePlacement
    print("Defining Summary Writer...")
    summary_writer = tf.summary.FileWriter(LogDir, sess.graph)
    max_acc = 0
    try:
        while not coord.should_stop():
            batch_x, batch_y = sess.run([X_train, y_train])
            print("batch_y: \n", batch_y)
            _, Acc, pre_y = sess.run([train_op, acc, y_op], feed_dict={x: batch_x, y_: batch_y})
            print("pre_y:\n", pre_y)
            if epoch_i % 10 == 0:
                if max_acc < Acc:
                    max_acc = Acc
                with open("Records/train_records.txt", "a") as file:
                    file.write(str("%d\t%.6f\t%.6f\n") % (epoch_i + 1, Acc, max_acc))

            if ((epoch_i + 1) % 10 == 0) or ((epoch_i + 1) == n_epoch):
                summary_str = sess.run(summ_op)
                summary_writer.add_summary(summary_str, epoch_i + 1)
                chckpoint_path = os.path.join(LogDir, 'model.ckpt')
                print("saving checkpoint into %s-%s" % (chckpoint_path, epoch_i + 1))
                saver.save(sess, chckpoint_path, global_step=epoch_i + 1)

            print("epoch_i: ", epoch_i, "   Acc: ", Acc)
            epoch_i = epoch_i + 1
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")
    finally:
        coord.request_stop()

coord.join(threads)

sess.close()
print("ok")

def SegAndSaveImages(saver, raw_image, seg_image):
    with tf.Session(config=tf.ConfigProto(log_device_placement=PARAMS.log_device_placement)) as sess:
        ckpt = tf.train.get_checkpoint_state(PARAMS.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint file found!')
            return

        # start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            # calculate the number of iterations for validation set. Because the batch size in this case
            # is 1, we just need to excute TOTAL_VALID_EXAMPLES times loop.
            step = 0
            while not coord.should_stop() and step < TOTAL_VALID_IMAGES:
                print('processing image %d/%d...' % (step + 1, TOTAL_VALID_IMAGES))

                image_raw, image_seg = sess.run([raw_image, seg_image])

                image_name = rawNameList[step].split('/')[-1]
                image_path = os.path.join(SAVE_IMAGE_DIR, image_name)

                misc.imsave(image_path, image_seg[0, :, :, 0])

                step += 1
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def GetBatch(rawImageList, BatchSize = 1):
    rawImageList = tf.cast(rawImageList, tf.string, name='CastrawFileName')

    NUM_EXAMPLES = rawImageList.shape[0].value
    print("Total Validation Examples: %d" % NUM_EXAMPLES)

    # Make an input queue
    InputQueue = tf.train.slice_input_producer([rawImageList],
                                               num_epochs=None,
                                               shuffle=False,
                                               capacity=16,
                                               shared_name=None,
                                               name='SliceInputProducer')

    # Read one example from input queue
    rawImageContent = tf.read_file(InputQueue[0], name='ReadrawImage')


    # Decode the jpeg image format
    rawImage = tf.image.decode_image(rawImageContent, channels=1, name='DecodeRawImage')

    with tf.name_scope('SetShape'):
        rawImage.set_shape([WIDTH, HEIGHT, 1])

    rawImageBatch = tf.train.batch([rawImage],
                                                  batch_size=BatchSize,
                                                  name='SuffleBatch')


    # Cast to tf.float32
    with tf.name_scope('CastToFloat'):
        rawImageBatch = tf.cast(rawImageBatch, tf.float32)

    # Normalization
    with tf.name_scope('Normalization'):
        rawImageBatch = rawImageBatch / 255.0

    return rawImageBatch

def classify(file_dir):
    with tf.Graph().as_default() as g:

        raw_image = GetBatch(rawNameList)

        # Build computational graph
        seg_image = unet_model.UNet(raw_image)

        saver = tf.train.Saver()
        #summ_op = tf.summary.merge_all()
        #summ_writer = tf.summary.FileWriter(PARAMS.val_log_dir, g)
        while True:
            SegAndSaveImages(saver, raw_image, seg_image)
            break


if __name__ == '__main__':
   # tf.app.run()
    classify(file_dir)

