
import unittest

import DataProcessing_2D
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal, assert_raises, assert_array_compare

A_df = pd.DataFrame(data={'A': [1, 2, 7],
                        'Exclude': [0, 0, 1],
                        'ID': ["blbl", "4blbl", "9m"]})

class TestData(unittest.TestCase):


    def test_1_data_processing_multi(self):
        XY = DataProcessing_2D.data_processing()

        self.assertEqual(len(XY[0]), len(XY[1]), True)
        self.assertEqual(XY[0].shape[1], 42*52, True)

        # Uncomment this case if number of images in a database is not 1075
        self.assertEqual(XY[0].shape[0], 1075*2, True)

        self.assertEqual(XY[1].max(), 2)
        self.assertEqual(XY[1].min(), 0)

        XY = DataProcessing_2D.data_processing(True, "max")

        self.assertEqual(len(XY[0]), len(XY[1]), True)
        self.assertEqual(XY[0].shape[1], 42*52, True)
        self.assertEqual(XY[1].shape[0], 1075*2, True)

        # Uncomment this case if number of images in a database is not 1075
        self.assertEqual(XY[1].shape[0], 1075*2, True)

        # multi class classificaitons:0,1,2
        self.assertEqual(XY[1].max(), 2)
        self.assertEqual(XY[1].min(), 0)

        # Min value should be zero in an image
        self.assertEqual(XY[0].min(), 0)

        print("Data processing for multi-class is complete")


    def test_2_data_processing_binary(self):
        XYA = DataProcessing_2D.data_processing(True, "std", "AD")
        XYB = DataProcessing_2D.data_processing(True, "mean", "MCI")
        XYC = DataProcessing_2D.data_processing(True, "max", "CN")

        self.assertEqual(len(XYA[0]), len(XYA[1]), True)
        self.assertEqual(len(XYB[0]), len(XYB[1]), True)
        self.assertEqual(len(XYC[0]), len(XYC[1]), True)

        lenA = len(XYA[0])
        lenB = len(XYB[0])
        lenC = len(XYC[0])

        # since each two binary classes excludes one class,
        # so total should be len of data len(labels)
        total = (2150-lenA) + (2150-lenB) + (2150-lenC)

        self.assertEqual(total, 1075*2)

        self.assertEqual(XYA[1].max(), 1)
        self.assertEqual(XYB[1].max(), 1)
        self.assertEqual(XYC[1].max(), 1)

        self.assertEqual(XYA[1].min(), 0)
        self.assertEqual(XYB[1].min(), 0)
        self.assertEqual(XYC[1].min(), 0)

        # Maximum projection should always be more than mean or std, with 0 lowest value
        self.assertEqual(XYA[0].sum()<XYC[0].sum(), True)
        self.assertEqual(XYB[0].sum()<XYC[0].sum(), True)

        print("Data processing for binary classification is complete")


if __name__=='__main__':
    unittest.main()
