"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import tensorflow3d as t3d
import numpy as np
import polyscope as ps

if __name__ == '__main__':
    ps.init()
    points = np.random.rand(100,3)
    pcd = t3d.representation.PointCloud()
    pcd.load("D:/research/tensorflow3d/tensorflow3d/tests/formats/ascii/xyz.txt")
    pcd.render()