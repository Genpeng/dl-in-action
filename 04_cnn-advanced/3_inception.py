# _*_ coding: utf-8 _*_

"""
A simplified TensorFlow implementation of GoogleNet (Inception v1).

Result: the accuracy on the test set is 0.72900 (train 10k steps)

Author: Genpeng Xu
Date:   2019/03/30
"""

import os
import numpy as np
import tensorflow as tf
from time import time
from dataset.cifar import CifarData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

CIFAR_DIR = "../data/cifar-10-batches-py/"


def inception_block(X, output_channel_sizes, name):
    """Build a Inception block."""
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(inputs=X,
                                   filters=output_channel_sizes[0],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv1_1')  # convolution layer with 1 * 1 kernel size
        conv3_3 = tf.layers.conv2d(inputs=X,
                                   filters=output_channel_sizes[1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv3_3')  # convolution layer with 3 * 3 kernel size
        conv5_5 = tf.layers.conv2d(inputs=X,
                                   filters=output_channel_sizes[2],
                                   kernel_size=(5, 5),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv5_5')  # convolution layer with 5 * 5 kernel size
        pool = tf.layers.max_pooling2d(inputs=X,
                                       pool_size=(2, 2),
                                       strides=(2, 2),
                                       name='pool')
        input_shape = X.get_shape().as_list()[1:]
        pool_shape = pool.get_shape().as_list()[1:]
        width_padding = (input_shape[0] - pool_shape[0]) // 2
        height_padding = (input_shape[1] - pool_shape[1]) // 2
        pool_padded = tf.pad(pool,
                             paddings=[[0, 0],
                                       [width_padding, width_padding],
                                       [height_padding, height_padding],
                                       [0, 0]])
        output = tf.concat([conv1_1, conv3_3, conv5_5, pool_padded], axis=3)
    return output


def main():
    # Assemble a graph
    # ========================================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name='X')  # (None, 3072)
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')  # (None, )

    with tf.name_scope('transform'):
        X_reshaped = tf.reshape(X, shape=[-1, 3, 32, 32], name='X_reshaped')
        X_transposed = tf.transpose(X_reshaped, perm=[0, 2, 3, 1], name='X_transposed')  # 3 @ 32 * 32

    with tf.variable_scope('ConvPool_1'):
        conv1 = tf.layers.conv2d(inputs=X_transposed,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1')  # 32 @ 32 * 32
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool1')  # 32 @ 16 * 16

    # inception block 1
    inception_2a = inception_block(pool1, [16, 16, 16], name='inception_2a')  # 80 @ 16 * 16

    # inception block 2
    inception_2b = inception_block(inception_2a, [16, 16, 16], name='inception_2b')  # 80 @ 16 * 16

    with tf.variable_scope('pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=inception_2b,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool2')  # 80 @ 8 * 8

    # inception block 3
    inception_3a = inception_block(pool2, [16, 16, 16], name='inception_3a')  # 80 @ 8 * 8

    # inception block 4
    inception_3b = inception_block(inception_3a, [16, 16, 16], name='inception_3b')  # 80 @ 8 * 8

    with tf.variable_scope('pool3'):
        pool3 = tf.layers.max_pooling2d(inputs=inception_3b,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool3')  # 80 @ 4 * 4

    with tf.variable_scope('output'):
        pool3_flattened = tf.layers.flatten(pool3)
        logits = tf.layers.dense(inputs=pool3_flattened, units=10)

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

    # Use session to execute the graph
    # ========================================================================================== #

    print("[INFO] Start training...")
    print()
    t0 = time()

    batch_size = 20
    train_steps = 10000
    test_steps = 500

    train_filepaths = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filepaths = [os.path.join(CIFAR_DIR, 'test_batch')]
    train_data, test_data = CifarData(train_filepaths), None

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("./graph/3_inception/", graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, train_steps + 1):
            batch_data_train, batch_labels_train = train_data.next_batch(batch_size)
            loss_train, acc_train, _ = sess.run(fetches=[loss, accuracy, train_op],
                                                feed_dict={X: batch_data_train, y: batch_labels_train})
            if i % 100 == 0:
                print("[Train] Step: %6d, loss: %4.5f, acc: %4.5f" % (i, loss_train, acc_train))
            if i % 1000 == 0:
                test_data = CifarData(test_filepaths)
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
    writer.close()

    print()
    print("[INFO] Traning finished! ( ^ _ ^ ) V")
    print("[INFO] Done in %f seconds." % (time() - t0))


if __name__ == '__main__':
    main()
