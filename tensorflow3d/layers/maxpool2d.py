"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class MaxPool2D(tf.keras.layers.Layer):

    def __init__(self, name, strides=[1, 1, 1, 1], padding='VALID'):
        super(MaxPool2D, self).__init__()
        self.shape = None
        self.id = name
        self.padding = padding
        self.strides = strides

    def build(self, input_shape):
        self.shape = [1, input_shape[1], 1, 1]

    def call(self, inputs):
        return tf.math.reduce_max(inputs, axis=1, keepdims=True)
        # return tf.nn.max_pool2d(inputs, ksize=self.shape, name=self.id,
        #                         strides=self.strides, padding=self.padding)
