"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import unittest
from tensorflow3d.io import Loader

class ReadPointCloudTestCase(unittest.TestCase):

    """
        :ops: Test cases for all point cloud functionalities
    """
    def test_ascii_file(self):
        print("...Testing point cloud ascii formats")
        file = Loader()

        for format in file.format_value_list:
            if format != 'auto':
                print(f"   *** Result for {format}: ", end='')
                try:
                    result = file.read_obj(path=f"tensorflow3d/tests/formats/ascii/{format}.txt", format={'point-cloud': format})
                    if result:
                        print("\u001b[1;32m PASSED\u001b[0m")
                    else:
                        print("\u001b[1;31m FAILED\u001b[0m")
                    self.assertEqual(result, True)
                except:
                    print("\u001b[1;31m FAILED\u001b[0m")