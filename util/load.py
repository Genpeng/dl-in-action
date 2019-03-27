# _*_ coding: utf-8 _*_

"""
Some utility functions about loading data.

Author: Genpeng Xu (xgp1227@gmail.com)
Data:   2019/03/25
"""

import pickle


def load_cifar_data(filepath):
    """Load CIFAR-10 dataset from file and return samples and its corresponding labels."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


if __name__ == '__main__':
    # 测试 load_cifar_data
    import os
    cifar_dir = "../data/cifar-10-batches-py"
    data, labels = load_cifar_data(os.path.join(cifar_dir, 'data_batch_1'))
    print(data[:2])
    print(labels[:2])
