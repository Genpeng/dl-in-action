# _*_ coding: utf-8 _*_

"""
A simple example about some common data augmentation APIs.

Author: Genpeng Xu
Date:   2019/04/07
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_PATH = "../data/gugong.jpg"


def test():
    img_str = tf.read_file(IMG_PATH)
    img_decoded = tf.image.decode_image(img_str)
    with tf.Session() as sess:
        img_decoded_array = sess.run(img_decoded)
    print(img_decoded_array.shape)
    plt.imshow(img_decoded_array)
    plt.show()


def resize_api():
    """
    Some frequently-used APIs about resizing are as follow:
    - tf.image.resize_area
    - tf.image.resize_bicubic
    - tf.image.resize_nearest_neighbor
    """
    img_str = tf.read_file(IMG_PATH)
    img_decoded = tf.image.decode_image(img_str)
    img_reshaped = tf.reshape(img_decoded, shape=[-1, 365, 600, 3])
    img_resized = tf.image.resize_bicubic(img_reshaped, size=[730, 1200])
    with tf.Session() as sess:
        img_resized_arr = sess.run(img_resized)
        img_resized_arr = img_resized_arr.reshape((730, 1200, 3)).astype(np.uint8)
        print(img_resized_arr.shape)
    plt.imshow(img_resized_arr)
    plt.show()


def crop_api():
    """
    Some frequently-used APIs about cropping are as follow:
    - tf.image.pad_to_bounding_box
    - tf.image.crop_to_bounding_box
    - tf.random_drop
    """
    img_str = tf.read_file(IMG_PATH)
    img_decoded = tf.image.decode_image(img_str)
    img_reshaped = tf.reshape(img_decoded, shape=[-1, 365, 600, 3])
    img_padded = tf.image.pad_to_bounding_box(img_reshaped, 50, 100, 500, 800)
    with tf.Session() as sess:
        img_padded_arr = sess.run(img_padded)
        img_padded_arr = img_padded_arr.reshape((500, 800, 3)).astype(np.uint8)
        print(img_padded_arr.shape)
    plt.imshow(img_padded_arr)
    plt.show()


def flip_api():
    """
    Some frequently-used APIs about flipping are as follow:
    - tf.image.flip_up_down
    - tf.image.flip_left_right
    - tf.random_flip_up_down
    - tf.random_flip_left_right
    """
    img_str = tf.read_file(IMG_PATH)
    img_decoded = tf.image.decode_image(img_str)
    img_reshaped = tf.reshape(img_decoded, shape=[-1, 365, 600, 3])
    img_flipped = tf.image.flip_left_right(img_reshaped)
    with tf.Session() as sess:
        img_flipped_arr = sess.run(img_flipped)
        img_flipped_arr = img_flipped_arr.reshape((365, 600, 3)).astype(np.uint8)
        print(img_flipped_arr.shape)
    plt.imshow(img_flipped_arr)
    plt.show()


def brightness_api():
    """
    Some frequently-used APIs about changing brightness are as follow:
    - tf.image.adjust_brightness
    - tf.image.random_brightness
    - tf.image.adjust_constrast
    - tf.image.random_constrast
    """
    img_str = tf.read_file(IMG_PATH)
    img_decoded = tf.image.decode_image(img_str)
    img_reshaped = tf.reshape(img_decoded, shape=[-1, 365, 600, 3])
    img_adjusted = tf.image.adjust_brightness(img_reshaped, 0.5)
    with tf.Session() as sess:
        img_adjusted_arr = sess.run(img_adjusted)
        img_adjusted_arr = img_adjusted_arr.reshape((365, 600, 3)).astype(np.uint8)
        print(img_adjusted_arr.shape)
    plt.imshow(img_adjusted_arr)
    plt.show()


if __name__ == '__main__':
    test()
    resize_api()
    crop_api()
    flip_api()
    brightness_api()
