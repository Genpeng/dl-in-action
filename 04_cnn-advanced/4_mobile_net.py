# _*_ coding: utf-8 _*_

"""
A simplified TensorFlow implementation of MobileNet.

Result: the accuracy on the test set is 0.63100 (train 10k steps)

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


def separable_conv_block(X, num_output_channel, name):
    """Build a deep separable convolution block."""
    with tf.variable_scope(name):
        num_input_channel = X.get_shape().as_list()[-1]
        Xs_channel_wise = tf.split(X, num_input_channel, axis=3)
        sepa_conv_outputs = []
        for i in range(len(Xs_channel_wise)):
            sepa_conv_output = tf.layers.conv2d(inputs=Xs_channel_wise[i],
                                                filters=1,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding='same',
                                                activation=tf.nn.relu,
                                                name='sepa_conv_%d' % i)
            sepa_conv_outputs.append(sepa_conv_output)
        sepa_conv_output_merged = tf.concat(sepa_conv_outputs, axis=3)
        conv1_1 = tf.layers.conv2d(inputs=sepa_conv_output_merged,
                                   filters=num_output_channel,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv1_1')
        return conv1_1


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

    # separable convolution block 1
    separable_2a = separable_conv_block(pool1, 32, name='separable_2a')  # 32 @ 16 * 16

    # separable convolution block 2
    separable_2b = separable_conv_block(separable_2a, 32, name='separable_2b')  # 32 @ 16 * 16

    with tf.variable_scope('pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=separable_2b,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool2')  # 32 @ 8 * 8

    # separable convolution block 3
    separable_3a = separable_conv_block(pool2, 32, name='separable_3a')  # 32 @ 8 * 8

    # separable convolution block 4
    separable_3b = separable_conv_block(separable_3a, 32, name='separable_3b')  # 32 @ 8 * 8

    with tf.variable_scope('pool3'):
        pool3 = tf.layers.max_pooling2d(inputs=separable_3b,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool3')  # 32 @ 4 * 4

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
    writer = tf.summary.FileWriter("./graph/4_mobile_net/", graph=tf.get_default_graph())
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
