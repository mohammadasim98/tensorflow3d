"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import polyscope as ps
import numpy as np
import tensorflow as tf


class Kitti():
    def __init__(self, path, samples, **kwargs):
        self.name = kwargs.get('name', None)
        self.path = path
        self.paths = tf.io.gfile.glob(path + '/*.npz')
        self.samples = samples

    def __call__(self, index):
        fn = self.paths[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pos1 = data['pos1']
            pos2 = data['pos2']
            flow = data['gt']

        # Sample from frame 1 and ground truth flow
        randi = np.random.randint(0, high=len(pos1), size=self.samples, dtype=int)
        pos1 = pos1[randi]
        flow = flow[randi]

        # Sample from frame 2
        randi = np.random.randint(0, high=len(pos2), size=self.samples, dtype=int)
        pos2 = pos2[randi]

        color1 = np.zeros([self.samples, 3])
        color2 = np.zeros([self.samples, 3])
        mask = np.ones([self.samples])

        return pos1, pos2, color1, color2, flow, mask

    def __len__(self):
        return len(self.path)

    def generator(self):
        for index in range(len(self)):
            yield self(index)

    def build(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(tf.TensorSpec(shape=(self.samples, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(self.samples, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(self.samples, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(self.samples, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(self.samples, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(self.samples,), dtype=tf.float32))
        )

    def render(self, name):
        """
        @ops: Render the point cloud including color, intensity value and surface normals
        @args:
            name: Name to the rendered plot
                type: Str
        @return: None
        """

        ps.init()
        ps.set_up_dir("z_up")
        gen = self.generator()
        f1 = next(gen)
        ps_cloud = ps.register_point_cloud(name, f1[0])
        ps_cloud.set_radius(0.00042)

        for i in range(len(self)):
            frame = next(gen)
            ps_cloud.update_point_positions(frame[0])
            ps.show()
        del ps_cloud

