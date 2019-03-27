# _*_ coding: utf-8 _*_

import os
import pickle
import matplotlib.pyplot as plt

CIFAR_DIR = "../data/cifar-10-batches-py/"


def main():
    print(os.listdir(CIFAR_DIR))

    # 查看数据的情况
    with open(os.path.join(CIFAR_DIR, 'data_batch_1'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        print(type(data))
        print(data.keys())

        print(type(data[b'data']))
        print(type(data[b'labels']))
        print(type(data[b'batch_label']))
        print(type(data[b'filenames']))

        print(data[b'data'][:2])
        print(data[b'labels'][:2])
        print(data[b'batch_label'])
        print(data[b'filenames'][:2])

        # 显示其中一张图片
        image_arr = data[b'data'][100]
        image_arr = image_arr.reshape((3, 32, 32))
        image_arr = image_arr.transpose((1, 2, 0))
        plt.imshow(image_arr)
        plt.show()


if __name__ == '__main__':
    main()
