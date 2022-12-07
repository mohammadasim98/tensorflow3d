"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

    @reference: Chris Tralie
    @source: https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8

"""
import tensorflow as tf


class FPS(tf.keras.layers.Layer):
    """
    Transformation layer
    """

    def __init__(self, name, samples: int = 1024):
        """
        @ops: Initialize TNet parameters and layers
        @args:
            name: Unique name for the layer
                type: Str
            samples: Number of samples N_ required
                type: Int
        @return: None
        """
        super(FPS, self).__init__()
        self.sampled = None
        self.batch_size = None
        self.num_points = None
        self.id = name
        self.samples = samples

    @tf.function
    def call(self, inputs):
        """
        @ops: Call FPS layer on inputs
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxNx3
        @return: Output node of FPS
            type: KerasTensor
            shape: BxN_x3
        """
        # I:BxNx3 :: O:Bx1xNx3:
        t1 = tf.expand_dims(inputs, axis=1)
        # I:BxNx3 :: O:BxNx1x3:
        t2 = tf.expand_dims(inputs, axis=2)
        # I:[Bx1xNx3, BxNx1x3] :: O:BxNxN :: Compute Distance Matrix
        dist_matrix = tf.norm(t1 - t2, ord='euclidean', axis=3)

        # By @Chris Tralie but modified by @Mohammad Asim
        # I:BxNxN :: O:BxN :: Slice Distance Matrix
        ds = dist_matrix[:, 0, :]

        # I:B :: O:Bx1 :: Create Range on Batch Size
        ran = tf.range(tf.shape(inputs)[0], dtype=tf.int32)
        ran = tf.expand_dims(ran, axis=-1)
        # I:BxN :: O:Bx1 :: Compute Index
        idx = tf.math.argmax(ds, axis=-1, output_type=tf.int32)
        idx = tf.expand_dims(idx, axis=-1)
        # I:[Bx1, Bx1] :: O:Bx2 :: Concatenate Range and Index
        idx_concat = tf.concat([ran, idx], axis=-1)
        # I:[BxNx3, Bx2] :: O:Bx1x3 :: Get a sample
        self.sampled = tf.expand_dims(tf.gather_nd(inputs, idx_concat), axis=1)
        # I:BxNxN :: O:BxN :: Slice Distance Matrix
        new_ds = tf.gather_nd(dist_matrix, indices=idx_concat)
        # I:[BxN, BxN] :: O:BxN :: Get Minimum from Old and New Slices of Distance Matrix
        ds = tf.math.minimum(ds, new_ds)

        for i in range(1, self.samples):
            # I:BxN :: O:Bx1 :: Compute Index
            idx = tf.math.argmax(ds, axis=-1, output_type=tf.int32)
            idx = tf.expand_dims(idx, axis=-1)
            # I: [Bx1, Bx1]:: O: Bx2:: Concatenate Range and Index
            idx_concat = tf.concat([ran, idx], axis=-1)
            # I:[BxNx3, Bx2] :: O:BxTx3 :: Get new sample and concatenate with previous
            self.sampled = tf.concat([
                self.sampled,
                tf.expand_dims(tf.gather_nd(inputs, idx_concat), axis=1)
            ], axis=1)
            # I:BxNxN :: O:BxN :: Slice Distance Matrix
            new_ds = tf.gather_nd(dist_matrix, indices=idx_concat)
            # I:[BxN, BxN] :: O:BxN :: Get Minimum from Old and New Slices of Distance Matrix
            ds = tf.math.minimum(ds, new_ds)

        return self.sampled
