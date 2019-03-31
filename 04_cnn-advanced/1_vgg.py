# _*_ coding: utf-8 _*_

"""
A simplified TensorFlow implementation of VGG net, where the core idea of VGG is
to use 3 * 3 kernel instead of 5 * 5 kernel or other kernels of bigger size, and
that can make the network to become deeper.

Result: the accuracy on the test set is 0.71190 (train 10k steps)

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


def main():
    # Assemble a graph
    # ========================================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name='X')  # (None, 3072)
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')  # (None, )

    with tf.name_scope('transform'):
        X_reshaped = tf.reshape(X, shape=[-1, 3, 32, 32], name='X_reshaped')
        X_transposed = tf.transpose(X_reshaped, perm=[0, 2, 3, 1], name='X_transposed')  # 3 @ 32 * 32

    # two convolution block 1
    conv1_1 = tf.layers.conv2d(inputs=X_transposed,
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

    with tf.name_scope("FC"):
        pool3_flattened = tf.layers.flatten(pool3, name='pool3_flattened')
        Z10 = tf.layers.dense(pool3_flattened, units=10, name='Z10')  # Z10 is the input of output layer

    # loss
    with tf.name_scope('loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=Z10)

    # metric
    with tf.name_scope('accuracy'):
        preds = tf.argmax(Z10, axis=1)  # (None, )
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
    writer = tf.summary.FileWriter("./graph/1_vgg/", graph=tf.get_default_graph())
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
