"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import tensorflow3d as t3d
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


if __name__ == '__main__':
    # inputs = tf.keras.Input(shape=(256, 3))
    pnet = t3d.models.PointNet(name='pointnet').build_graph(input_shape=(1024, 3))
    # model = tf.keras.models.Model(inputs=inputs, outputs=pnet)
    pnet.summary()


