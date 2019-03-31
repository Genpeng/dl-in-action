# _*_ coding: utf-8 _*_

"""
A simplified TensorFlow implementation of ResNet.

Result: the accuracy on the test set is 0.75350 (train 10k steps)

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


def residual_block(X, output_channel):
    """Build residual block."""
    input_channel = X.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception("[ERROR] Input channel can not match output channel!!!")
    conv1 = tf.layers.conv2d(inputs=X,
                             filters=output_channel,
                             kernel_size=(3, 3),
                             strides=strides,
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv1')
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=output_channel,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv2')
    if increase_dim:
        X_pooled = tf.layers.average_pooling2d(inputs=X,
                                               pool_size=(2, 2),
                                               strides=(2, 2),
                                               padding='valid')
        X_padded = tf.pad(X_pooled,
                          paddings=[[0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [input_channel // 2, input_channel // 2]])
    else:
        X_padded = X
    return conv2 + X_padded


def build_resnet(X, num_residual_blocks, num_filter_base, num_class):
    """Build a deep residual neural network (ResNet)."""
    num_subsampling = len(num_residual_blocks)
    layers = []
    with tf.variable_scope("conv0"):
        conv0 = tf.layers.conv2d(inputs=X,
                                 filters=num_filter_base,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv0')
        layers.append(conv0)
        # The original ResNet also has a max-pooling layer after the first convolution layer,
        # but here we don't implement it
    for block_idx in range(num_subsampling):
        for i in range(num_residual_blocks[block_idx]):
            with tf.variable_scope("ResBlock%d_%d" % (block_idx + 1, i + 1)):
                conv = residual_block(layers[-1], output_channel=num_filter_base * (2 ** block_idx))
                layers.append(conv)
    # validate the output size
    input_size = X.get_shape().as_list()[1:]
    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] == [input_size[0] / multiplier,
                                                    input_size[1] / multiplier,
                                                    num_filter_base * multiplier]
    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], axis=[1, 2])  # (None, channel)
        logits = tf.layers.dense(global_pool, num_class)
        layers.append(logits)
    return layers[-1]


def main():
    # Assemble a graph
    # ========================================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name='X')  # (None, 3072)
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')  # (None, )

    with tf.name_scope('transform'):
        X_reshaped = tf.reshape(X, shape=[-1, 3, 32, 32], name='X_reshaped')
        X_transposed = tf.transpose(X_reshaped, perm=[0, 2, 3, 1], name='X_transposed')  # 3 @ 32 * 32

    logits = build_resnet(X_transposed, [2, 3, 2], 32, 10)

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
    writer = tf.summary.FileWriter("./graph/2_resnet/", graph=tf.get_default_graph())
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
