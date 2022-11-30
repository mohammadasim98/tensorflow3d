"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import tensorflow3d as t3d

if __name__ == '__main__':
    pcd = t3d.representation.PointCloud()
    pcd.load("D:/research/tensorflow3d/tensorflow3d/tests/formats/ascii/xyz.txt", format='xyzi')
    pcd.render()