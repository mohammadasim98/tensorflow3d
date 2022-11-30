"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import polyscope as ps
import numpy as np

class PointCloud():
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.points = kwargs.get('points', [])
        self.colors = kwargs.get('colors', [])
        self.values = kwargs.get('values', [])

    def load(self, path, format: str='xyz'):
        with open(path) as file:
            try:
                line = file.readline()
                line_length = len(line)
                self.points = []
                self.values = []
                self.colors = []
                while line:
                    line = line.split(" ")
                    line = list(np.array(line).astype(np.float))
                    if format == 'xyzi':
                        self.values.append(line[-1])
                    elif format == 'xyzrgb':
                        self.colors.append(line[3:])
                    self.points.append(line[:3])
                    line = file.readline()

                self.points = np.array(self.points)
                self.values = np.array(self.values)
                self.colors = np.array(self.colors)
                return True
            except:
                raise Exception('Unable to read file')
                return False

    def render(self, name: str='default'):
        ps.init()
        ps.set_up_dir("z_up")

        if self.points is not None or self.points != []:
            ps_cloud = ps.register_point_cloud(name, self.points)
            ps_cloud.set_radius(0.00042)

        if np.size(self.colors):
            ps_cloud.add_color_quantity(name, self.colors)

        if np.size(self.values):
            ps_cloud.add_scalar_quantity(name, self.values, enabled=True, cmap='turbo')

        ps.show()
        del ps_cloud

