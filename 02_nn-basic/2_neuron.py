# _*_ coding: utf-8 _*_

import os
import pickle
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

CIFAR_DIR = "../data/cifar-10-batches-py/"


def load_cifar_data(filepath):
    """Load CIFAR-10 dataset from file and return samples and its corresponding labels."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CifarData:
    """CIFAR-10 or CIFAR-100 dataset."""

    def __init__(self, filepaths, need_shuffle=True):
        all_data, all_labels = [], []
        for filepath in filepaths:
            data, labels = load_cifar_data(filepath)

            # all the categories
            # all_data.append(data)
            # all_labels.append(labels)

            # Only for binary classification
            for sample, label in zip(data, labels):  # sample: Series, label: int
                if label in [0, 1]:
                    all_data.append(sample)
                    all_labels.append(label)

        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1  # normalization
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

        # print("The shape of examples:", self._data.shape)
        # print("The shape of labels:", self._labels.shape)

    def _shuffle_data(self):
        indices_shuffled = np.random.permutation(self._num_examples)
        self._data = self._data[indices_shuffled]
        self._labels = self._labels[indices_shuffled]

    def next_batch(self, batch_size=128):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("Have no more examples!!!")
        if end_indicator > self._num_examples:
            raise Exception("The size of one batch is larger than the number of examples!!!")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


def main():
    # Assemble a graph
    # ========================================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name='X')
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')

    w = tf.get_variable(name='w', shape=[X.get_shape()[-1], 1], initializer=tf.random_normal_initializer(0, 1))
    b = tf.get_variable(name='b', shape=[1], initializer=tf.constant_initializer(0.0))

    # output
    z = tf.add(tf.matmul(X, w), b, name='z')  # (None, 1)
    probs = tf.nn.sigmoid(z, name='probs')  # (None, 1)

    # loss
    y_reshaped = tf.reshape(y, shape=(-1, 1))  # (None, 1)
    y_reshaped_float = tf.cast(y_reshaped, dtype=tf.float32)  # (None, 1)
    loss = tf.reduce_mean(tf.square(probs - y_reshaped_float), name='loss')  # o-d tensor

    # metric
    preds = tf.greater(probs, 0.5)
    correct_preds = tf.equal(tf.cast(preds, tf.int64), y_reshaped)
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Use session to execute the graph
    # ========================================================================================== #

    batch_size = 20
    train_steps = 100000
    test_steps = 100

    train_filepaths = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filepaths = [os.path.join(CIFAR_DIR, 'test_batch')]
    train_data, test_data = CifarData(train_filepaths), None

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("./graph/2_neuron/", graph=tf.get_default_graph())
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
