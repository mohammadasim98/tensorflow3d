"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
from tensorflow3d.layers.conv2d import Conv2D
from tensorflow3d.layers.maxpool2d import MaxPool2D
from tensorflow3d.layers.dense import Dense
from tensorflow3d.layers.tnet import TNet
import numpy as np


class PointNet(tf.keras.Model):
    """
    Spatial Transformation layer
    """

    def __init__(self, name):
        super(PointNet, self).__init__()
        self.num_points = None
        self.id = name
        self.batch_size = None

    def call(self, inputs):
        """
        Call Point Net layer on inputs
        @args: input point cloud
            :shape: BxNxL
        @return: output of point net
        """

        # BxNx3: Input Transformation
        net = TNet(name='input_transformation')(inputs)
        # BxNx3:
        net = tf.expand_dims(net, -1)
        # BxNx3x1:
        net = Conv2D(filters=64, shape=[1, 3], name=self.id + 'tconv2d_64_0')(net)
        net = Conv2D(filters=64, shape=[1, 1], name=self.id + 'tconv2d_64_1')(net)
        # BxNx1x64:
        net = tf.squeeze(net, axis=2)
        # BxNx64: Feature Transformation
        net = TNet(name='feature_transformation', k=64)(net)
        # BxNx64:
        net = tf.expand_dims(net, 2)
        # BxNx1x64:
        net = Conv2D(filters=64, shape=[1, 1], name=self.id + 'tconv2d_64_2')(net)
        net = Conv2D(filters=128, shape=[1, 1], name=self.id + 'tconv2d_128_0')(net)
        net = Conv2D(filters=1024, shape=[1, 1], name=self.id + 'tconv2d_1024_0')(net)
        # BxNx1x1024: Symmetric function
        net = MaxPool2D(name=self.id + 'tmaxpool2d')(net)
        # Bx1x1x1024:
        net = tf.keras.layers.Flatten()(net)
        # Bx1024:
        net = Dense(512, name=self.id + 'tdense_512')(net)
        # Bx512:
        net = tf.nn.dropout(net, 0.7, None)
        net = Dense(256, name=self.id + 'tdense_256')(net)
        # Bx256:
        net = tf.nn.dropout(net, 0.7, None)
        net = Dense(40, name=self.id + 'tdense_40')(net)
        # Bx40:
        return net

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
