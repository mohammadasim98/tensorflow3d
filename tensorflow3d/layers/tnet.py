"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
from .conv2d import Conv2D
from .maxpool2d import MaxPool2D
from .dense import Dense
import numpy as np


class TNet(tf.keras.layers.Layer):
    """
    Spatial Transformation layer
    """

    def __init__(self, name, kernel_initializer=tf.keras.initializers.RandomUniform(),
                 bias_initializer=tf.keras.initializers.Constant(0), expand=True, k=3):
        super(TNet, self).__init__()
        self.bias = None
        self.kernel = None
        self.num_points = None
        self.k = k
        self.id = name
        self.batch_size = None
        self.expand = expand
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.tconv2d_64 = Conv2D(filters=64, shape=[1, 3], name=self.id + '_tconv2d_64')
        self.tconv2d_128 = Conv2D(filters=128, shape=[1, 1], name=self.id + '_tconv2d_128')
        self.tconv2d_1024 = Conv2D(filters=1024, shape=[1, 1], name=self.id + '_tconv2d_1024')
        self.tdense_512 = Dense(512, name=self.id + '_tdense_512')
        self.tdense_256 = Dense(256, name=self.id + '_tdense_256')

    def build(self, input_shape):
        self.num_points = input_shape[1]
        self.batch_size = input_shape[0]
        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=(256, self.k**2), dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            initial_value=self.bias_initializer(shape=(self.k**2, ), dtype=tf.float32),
            trainable=True)

        self.bias.assign_add(tf.constant(np.eye(self.k).flatten(), dtype=tf.float32))

    def call(self, inputs):
        """
        Call transformation layer on inputs and features
        @args: input point cloud
            :shape: BxNxL
        @return: transformed point cloud
        """
        inp = inputs
        if self.expand:
            inp = tf.expand_dims(inputs, -1)

        net = self.tconv2d_64(inp)
        net = self.tconv2d_128(net)
        net = self.tconv2d_1024(net)
        net = MaxPool2D(name=self.id + '_tmaxpool2d')(net)

        net = tf.keras.layers.Flatten()(net)
        net = self.tdense_512(net)
        net = self.tdense_256(net)
        transform_mat = tf.matmul(net, self.kernel)
        transform_mat = tf.nn.bias_add(transform_mat, self.bias)
        transform_mat = tf.keras.layers.Reshape([self.k, self.k])(transform_mat)
        inputs_transformed = tf.matmul(inputs, transform_mat)
        return inputs_transformed
