"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class Dense(tf.keras.layers.Layer):

    def __init__(self, num_outputs, name):
        super(Dense, self).__init__()
        self.bias = None
        self.kernel = None
        self.num_outputs = num_outputs
        self.id = name

    def build(self, input_shape):
        kernel_shape = [input_shape[-1], self.num_outputs]
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer="random_normal",
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.num_outputs, ),
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=True)

    def call(self, inputs):
        net = tf.matmul(inputs, self.kernel)
        return tf.nn.bias_add(net, self.bias)

