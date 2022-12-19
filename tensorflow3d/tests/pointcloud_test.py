"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import unittest
from tensorflow3d.representation import PointCloud

class ReadPointCloudTestCase(unittest.TestCase):

    """
        :ops: Test cases for all point cloud functionalities
    """
    def test_ascii_file(self):
        print("...Testing point cloud ascii formats")
        pcd = PointCloud()

        for format in ['xyz', 'xyzi']:
            print(f"   *** Result for {format}: ", end='')
            try:
                result = pcd.load(f"tensorflow3d/tests/formats/ascii/{format}.txt", format=format)
                if result:
                    print("\u001b[1;32m PASSED\u001b[0m")
                else:
                    print("\u001b[1;31m FAILED\u001b[0m")
                self.assertEqual(result, True)
            except:
                print("\u001b[1;31m FAILED\u001b[0m")