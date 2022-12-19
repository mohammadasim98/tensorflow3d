"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import numpy as np


class Loader:
    def __init__(self):

        self.format_key_list = ['point-cloud']
        self.format_value_list = ['xyz', 'xyzi', 'auto']
        self.raw = None

    def verify(self, format):
        """
            :ops: Check if the given format is applicable
            :return: None
        """
        if list(format.keys())[-1] not in self.format_key_list:
            raise FormatError("Invalid format key")

        if list(format.values())[-1] not in self.format_value_list:
            raise FormatError("Invalid format value")
        return True

    def read_obj(self, path: str, format: dict):
        """
            :ops: Read input file and store raw data
            :return: bool
        """

        # Perform format verification
        self.verify(format)

        # Read file
        with open(path) as file:
            try:
                points = []
                while True:
                    line = file.readline()
                    if not line:
                        break
                    line = line.split(" ")
                    if (len(line) == 3):
                        xyz = list(np.array(line).astype(np.float))
                        points.append(xyz)
                    if (len(line) == 4):
                        xyz = list(np.array(line[:3]).astype(np.float))
                        points.append(xyz)
                points = np.array(points)
                print(points.shape)
                return True
            except:
                raise Exception('Unable to read file')
                return False


class FormatError(Exception):
    """ Raise exception on invalid format """
    pass
