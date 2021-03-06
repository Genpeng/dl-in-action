# _*_ coding: utf-8 _*_

"""
A simple example of the usage of tensorboard by showing the training procedure of VGG network.

In general, the steps of using tensorboard:
1. specify the variables to display
2. calculate the variables during training and write to file
3. run tensorboard command to parse the file

Author: Genpeng Xu
Date:   2019/04/06
"""

import os
import numpy as np
import tensorflow as tf
from time import time
from dataset.cifar import CifarData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

CIFAR_DIR = "../data/cifar-10-batches-py/"
LOG_DIR = "./runs"


def main():
    # Assemble a graph
    # ========================================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name='X')  # (None, 3072)
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')  # (None, )

    with tf.name_scope('transform'):
        X_reshaped = tf.reshape(X, shape=[-1, 3, 32, 32], name='X_reshaped')
        X_image = tf.transpose(X_reshaped, perm=[0, 2, 3, 1], name='X_image')  # 3 @ 32 * 32

    # two convolution block 1
    conv1_1 = tf.layers.conv2d(inputs=X_image,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv1_1')  # 32 @ 32 * 32
    conv1_2 = tf.layers.conv2d(inputs=conv1_1,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv1_2')  # 32 @ 32 * 32

    pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                    pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid',
                                    name='pool1')  # 32 @ 16 * 16

    # two convolution block 2
    conv2_1 = tf.layers.conv2d(inputs=pool1,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv2_1')  # 32 @ 16 * 16
    conv2_2 = tf.layers.conv2d(inputs=conv2_1,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv2_2')  # 32 @ 16 * 16

    pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                    pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid',
                                    name='pool2')  # 32 @ 8 * 8

    # two convolution block 3
    conv3_1 = tf.layers.conv2d(inputs=pool2,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv3_1')  # 32 @ 8 * 8
    conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv3_2')  # 32 @ 8 * 8

    pool3 = tf.layers.max_pooling2d(inputs=conv3_2,
                                    pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid',
                                    name='pool3')  # 32 @ 4 * 4

    with tf.name_scope("fc"):
        pool3_flattened = tf.layers.flatten(pool3, name='pool3_flattened')
        logits = tf.layers.dense(pool3_flattened, units=10, name='logits')  # logits is the input of output layer

    # loss
    with tf.name_scope('loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    # metric
    with tf.name_scope('accuracy'):
        preds = tf.argmax(logits, axis=1)  # (None, )
        correct_preds = tf.equal(preds, y)  # (None, )
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Specify variables to summary
    # ========================================================================================== #

    def variable_summary(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.histogram('histogram', var)

    with tf.name_scope('variables_summary'):
        variable_summary(conv1_1, 'conv1_1')
        variable_summary(conv1_2, 'conv1_2')
        variable_summary(conv2_1, 'conv2_1')
        variable_summary(conv2_2, 'conv2_2')
        variable_summary(conv3_1, 'conv3_1')
        variable_summary(conv3_2, 'conv3_2')

    with tf.name_scope('recover'):
        input_images = (X_image + 1) * 127.5
        tf.summary.image('input_images', input_images)

    loss_summary = tf.summary.scalar('loss', loss)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    train_summary = tf.summary.merge_all()
    test_summary = tf.summary.merge([loss_summary, accuracy_summary])

    # create log directory
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    run_label = "1_vgg_tensorboard"
    run_dir = os.path.join(LOG_DIR, run_label)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # create summary directory
    summary_dir = os.path.join(run_dir, 'summaries')
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    train_log_dir = os.path.join(summary_dir, 'train')
    test_log_dir = os.path.join(summary_dir, 'test')
    if not os.path.exists(train_log_dir):
        os.mkdir(train_log_dir)
    if not os.path.exists(test_log_dir):
        os.mkdir(test_log_dir)

    # Use session to execute the graph
    # ========================================================================================== #

    print("[INFO] Start training...")
    print()
    t0 = time()

    batch_size = 20
    train_steps = 10000
    test_steps = 500
    output_summary_every_steps = 100

    train_filepaths = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filepaths = [os.path.join(CIFAR_DIR, 'test_batch')]
    train_data, test_data = CifarData(train_filepaths), CifarData(test_filepaths, False)
    fixed_batch_data_test, fixed_batch_labels_test = test_data.next_batch(batch_size)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)

        for i in range(1, train_steps + 1):
            batch_data_train, batch_labels_train = train_data.next_batch(batch_size)
            train_ops = [loss, accuracy, train_op]
            should_output_summary = i % output_summary_every_steps == 0
            if should_output_summary:
                train_ops.append(train_summary)

            train_results = sess.run(fetches=train_ops,
                                     feed_dict={X: batch_data_train,
                                                y: batch_labels_train})
            loss_train, acc_train = train_results[:2]
            if should_output_summary:
                train_summary_str = train_results[-1]
                train_writer.add_summary(train_summary_str, i)
                test_summary_str = sess.run(fetches=[test_summary],
                                            feed_dict={X: fixed_batch_data_test,
                                                       y: fixed_batch_labels_test})[0]
                test_writer.add_summary(test_summary_str, i)

            if i % 100 == 0:
                print("[Train] Step: %6d, loss: %4.5f, acc: %4.5f" % (i, loss_train, acc_train))
            if i % 1000 == 0:
                test_data = CifarData(test_filepaths, False)
                losses_test, accs_test = [], []
                for j in range(test_steps):
                    batch_data_test, batch_labels_test = test_data.next_batch(batch_size)
                    loss_test, acc_test = sess.run(fetches=[loss, accuracy],
                                                   feed_dict={X: batch_data_test, y: batch_labels_test})
                    losses_test.append(loss_test)
                    accs_test.append(acc_test)
                loss_test = np.mean(losses_test)
                acc_test = np.mean(accs_test)
                print()
                print("[Test ] Step: %6d, loss: %4.5f, acc: %4.5f" % (i, loss_test, acc_test))
                print()
    train_writer.close()
    test_writer.close()

    print()
    print("[INFO] Training finished! ( ^ _ ^ ) V")
    print("[INFO] Done in %f seconds." % (time() - t0))


if __name__ == '__main__':
    main()
