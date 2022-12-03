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

    def __init__(self, name, expand=True, k=3):
        super(TNet, self).__init__()
        self.bias = None
        self.kernel = None
        self.num_points = None
        self.k = k
        self.id = name
        self.batch_size = None
        self.expand = expand

    def build(self, input_shape):
        self.num_points = input_shape[1]
        self.batch_size = input_shape[0]
        self.kernel = self.add_weight(shape=[256, self.k * self.k],
                                      initializer="random_normal",
                                      trainable=True)
        self.bias = self.add_weight(shape=[self.k * self.k],
                                    initializer=tf.constant_initializer(0.0),
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

        net = Conv2D(filters=64, shape=[1, 3], name=self.id + 'tconv2d_64')(inp)
        net = Conv2D(filters=128, shape=[1, 1], name=self.id + 'tconv2d_128')(net)
        net = Conv2D(filters=1024, shape=[1, 1], name=self.id + 'tconv2d_1024')(net)
        net = MaxPool2D(name=self.id + 'tmaxpool2d')(net)

        net = tf.keras.layers.Flatten()(net)
        net = Dense(512, name=self.id + 'tdense_512')(net)
        net = Dense(256, name=self.id + 'tdense_256')(net)
        transform_mat = tf.matmul(net, self.kernel)
        transform_mat = tf.nn.bias_add(transform_mat, self.bias)
        transform_mat = tf.keras.layers.Reshape([self.k, self.k])(transform_mat)
        inputs_transformed = tf.matmul(inputs, transform_mat)
        return inputs_transformed
