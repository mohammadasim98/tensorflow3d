"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import tensorflow3d as t3d
import tensorflow as tf


if __name__ == '__main__':

    # inputs = tf.keras.Input(shape=(256, 3))
    # pnet = t3d.models.PointNet(name='pointnet').build(input_shape=(None, 3))
    # model = tf.keras.models.Model(inputs=inputs, outputs=pnet)

    # pnet.summary()

    pcd = t3d.representation.PointCloud()
    pcd = pcd.load('tensorflow3d/tests/formats/ascii/xyzi.txt', mode='xyzi')
    #
    # print(pcd.shape)
    # pcd.sample(6912)
    # pnet2 = t3d.models.PointNet2(name='pointnet++').build(input_shape=(None, 3))
    #
    # print("Inferencing")
    # result = pnet2(tf.expand_dims(pcd.points, axis=0))
    # pcd.points = result.numpy()[0, ...]
    pcd.render(values=True)
