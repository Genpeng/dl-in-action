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

    A1 = tf.layers.dense(X, units=100, activation=tf.nn.relu, name='A1')
    A2 = tf.layers.dense(A1, units=100, activation=tf.nn.relu, name='A2')
    A3 = tf.layers.dense(A2, units=50, activation=tf.nn.relu, name='A3')

    Z4 = tf.layers.dense(A3, units=10, name='Z4')  # Z4 is the input of output layer

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
    writer = tf.summary.FileWriter("./graph/4_dnn/", graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, train_steps + 1):
            batch_data_train, batch_labels_train = train_data.next_batch(batch_size)
            loss_train, acc_train, _ = sess.run(fetches=[loss, accuracy, train_op],
                                                feed_dict={X: batch_data_train, y: batch_labels_train})
            if i % 500 == 0:
                print("[Train] Step: %6d, loss: %4.5f, acc: %4.5f" % (i, loss_train, acc_train))
            if i % 5000 == 0:
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
