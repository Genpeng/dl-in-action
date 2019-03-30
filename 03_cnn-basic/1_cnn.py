# _*_ coding: utf-8 _*_

import os
import numpy as np
import tensorflow as tf
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
        X_transposed = tf.transpose(X_reshaped, perm=[0, 2, 3, 1], name='X_transposed')

    with tf.name_scope("ConvPooling-1"):
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

    with tf.name_scope("ConvPooling-2"):
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv2')  # 32 @ 16 * 16
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool2')  # 32 @ 8 * 8

    with tf.name_scope("ConvPooling-3"):
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv3')  # 32 @ 8 * 8
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid',
                                        name='pool3')  # 32 @ 4 * 4

    with tf.name_scope('fc'):
        pool3_flattened = tf.layers.flatten(pool3, name='pool3_flattened')
        Z4 = tf.layers.dense(pool3_flattened, units=10, name='Z4')  # Z4 is the input of output layer

    # loss
    with tf.name_scope('loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=Z4)

    # metric
    with tf.name_scope('accuracy'):
        preds = tf.argmax(Z4, axis=1)  # (None, )
        correct_preds = tf.equal(preds, y)  # (None, )
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Use session to execute the graph
    # ========================================================================================== #

    batch_size = 20
    train_steps = 10000
    test_steps = 500

    train_filepaths = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filepaths = [os.path.join(CIFAR_DIR, 'test_batch')]
    train_data, test_data = CifarData(train_filepaths), None

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("./graph/1_cnn/", graph=tf.get_default_graph())
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


if __name__ == '__main__':
    main()
