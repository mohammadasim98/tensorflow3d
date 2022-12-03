"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):

    def __init__(self, filters, shape, name, kernel_initializer=tf.keras.initializers.RandomNormal(),
                 strides=[1, 1, 1, 1], padding='VALID'):
        super(Conv2D, self).__init__()
        self.bias = None
        self.kernel = None
        self.id = name
        self.shape = shape
        self.padding = padding
        self.strides = strides
        self.filters = filters
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        kernel_shape = [self.shape[0], self.shape[1], input_shape[-1], self.filters]
        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=kernel_shape, dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
        return tf.nn.conv2d(inputs, filters=self.kernel, name=self.id,
                            strides=self.strides, padding=self.padding)
