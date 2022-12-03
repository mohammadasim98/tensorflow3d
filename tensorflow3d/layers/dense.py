"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class Dense(tf.keras.layers.Layer):

    def __init__(self, num_outputs, name, kernel_initializer=tf.keras.initializers.RandomUniform(),
                 bias_initializer=tf.keras.initializers.Constant(0)):
        super(Dense, self).__init__()
        self.bias = None
        self.kernel = None
        self.num_outputs = num_outputs
        self.id = name
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        kernel_shape = [input_shape[-1], self.num_outputs]
        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=kernel_shape, dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            initial_value=self.bias_initializer(shape=(self.num_outputs, ), dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
        net = tf.matmul(inputs, self.kernel)
        return tf.nn.bias_add(net, self.bias)

