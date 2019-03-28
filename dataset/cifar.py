# _*_ coding: utf-8 _*_

"""
A utility class for loading CIFAR-10 or CIFAR-100 dataset.

References:
1. https://www.cs.toronto.edu/~kriz/cifar.html
2. https://coding.imooc.com/learn/list/259.html

Author: Genpeng Xu (xgp1227@gmail.com)
Date:   2019/03/25
"""

import numpy as np
from util.load import load_cifar_data


class CifarData:
    """CIFAR-10 or CIFAR-100 dataset."""

    def __init__(self, filepaths, need_shuffle=True):
        all_data, all_labels = [], []
        for filepath in filepaths:
            data, labels = load_cifar_data(filepath)
            all_data.append(data)
            all_labels.append(labels)
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
                raise Exception("There have no more examples!!!")
        if end_indicator > self._num_examples:
            raise Exception("The size of one batch is larger than the number of examples!!!")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


def main():
    # test CifarData class
    import os
    cifar_dir = "../data/cifar-10-batches-py/"
    train_filepaths = [os.path.join(cifar_dir, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filepaths = [os.path.join(cifar_dir, 'test_batch')]
    train_data = CifarData(train_filepaths)
    test_data = CifarData(test_filepaths, need_shuffle=False)
    batch_data_train, batch_labels_train = train_data.next_batch(5)
    batch_data_test, batch_labels_test = test_data.next_batch(5)
    print(batch_data_train)
    print(batch_labels_train)
    print(batch_data_test)
    print(batch_labels_test)


if __name__ == '__main__':
    main()
