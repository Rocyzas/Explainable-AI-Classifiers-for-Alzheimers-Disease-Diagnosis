
import unittest

import DataProcessing
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal, assert_raises, assert_array_compare

A_df = pd.DataFrame(data={'A': [1, 2, 7],
                        'Exclude': [0, 0, 1],
                        'ID': ["blbl", "4blbl", "9m"]})
A_df_exp = pd.DataFrame(data={'A': [1, 2],
                        'Exclude': [0, 0],
                        'ID': ["blbl", "4blbl"]})

B_df = pd.DataFrame(data={'A': [1, 2, 7],
                        'Exclude': [0, 0, 1],
                        'ID': ["blbl", "4blbl", "99"]})
B_df_exp = pd.DataFrame(data={'A': [1, 2],
                        'Exclude': [0, 0],
                        'ID': ["blbl", "4blbl"]})

C_df = pd.DataFrame(data={'A': [1.0, 3.0, np.nan],
                        'C': [0, 0, 0],
                        'ID': ["blbl", "4blbl", "99"]})
C_df_exp = pd.DataFrame(data={'A': [1.0, 3.0, 2.0],
                        'C': [0, 0, 0],
                        'ID': ["blbl", "4blbl", "99"]})

MERGEDA = pd.DataFrame(data={'ID': [0,1,2,3,4,5,6,7,8,9],
                        'MCI': [0,0,0,0,0,0,1,1,1,1],
                        'AD': [1,1,1,0,0,0,0,0,0,0],
                        'RightAmygdala': [-5,-8,8,6,0,1,3,-9,7,6],
                        'RightHippocampus': [5,-8,-8,66,0,15,30,-1,71,-6]})

MERGEDB = pd.DataFrame(data={'ID': range(0,40),
                        'MCI': [0,0,0,0,0,0,0,0,0,1]*4,
                        'AD': [1,0,0,0,0,0,0,0,0,0]*4,
                        'RightAmygdala': [0,0,0,0,0,0,0,0,0,0]*4,
                        'RightHippocampus': [5,-8,-8,66,0,15,30,-1,71,-6]*4})

class TestData(unittest.TestCase):

    def test_1_clear_data(self):
        # with 'm' in ID
        assert_frame_equal(DataProcessing.clear_data(A_df), A_df_exp)
        assert_frame_equal(DataProcessing.clear_data(A_df_exp), A_df_exp)

        # exclude
        assert_frame_equal(DataProcessing.clear_data(B_df), B_df_exp)
        assert_frame_equal(DataProcessing.clear_data(B_df_exp), B_df_exp)

        # np nan value
        assert_frame_equal(DataProcessing.clear_data(C_df), C_df_exp)
        assert_frame_equal(DataProcessing.clear_data(C_df_exp), C_df_exp)

        print("Clear Data Tested")

    def test_2_getXY(self):
        XY = DataProcessing.getXY(MERGEDA, "HC_AD")
        self.assertEqual(np.array_equal(XY[1], [1,1,1,0,0,0]), True)
        self.assertEqual(np.array_equal(XY[1], [1,1,1,1,0,0]), False)

        XY = DataProcessing.getXY(MERGEDA, "MCI_AD")
        self.assertEqual(np.array_equal(XY[1], [1,1,1,0,0,0,0]), True)
        self.assertEqual(np.array_equal(XY[1], [1,1,1,1,0,0,0]), False)

        XY = DataProcessing.getXY(MERGEDA, "HC_MCI")
        self.assertEqual(np.array_equal(XY[1], [0,0,0,1,1,1,1]), True)
        self.assertEqual(np.array_equal(XY[1], [0,0,0,0,1,1,1]), False)

        self.assertEqual(np.array_equal(XY[2], ["RightAmygdala", "RightHippocampus"]), True)
        self.assertEqual(np.array_equal(XY[2], ["RightHippocampus", "RightAmygdala"]), False)
        self.assertEqual(np.array_equal(XY[4], MERGEDA), False)


        print("getXY with MERGED A tested")

    def test_3_getXY(self):
        XY = DataProcessing.getXY(MERGEDB, "HC_AD")
                                # 'MCI': [0,0,0,0,0,0,0,0,0,1]*4,
                                # 'AD': [1,0,0,0,0,0,0,0,0,0]*4,
        self.assertEqual(np.array_equal(XY[1], [1,0,0,0,0,0,0,0,0]*4), True)
        self.assertEqual(np.array_equal(XY[1], [0,0,0,0,0,0,0,0,0,0]*4), False)

        XY = DataProcessing.getXY(MERGEDB, "MCI_AD")
        self.assertEqual(np.array_equal(XY[1], [1,0]*4), True)
        self.assertEqual(np.array_equal(XY[1], [1,1]*4), False)

        XY = DataProcessing.getXY(MERGEDB, "HC_MCI")
        self.assertEqual(np.array_equal(XY[1], [0,0,0,0,0,0,0,0,1]*4), True)
        self.assertEqual(np.array_equal(XY[1], [1,0,0,0,0,0,0,0,0]*4), False)


        self.assertEqual(np.array_equal(XY[2], ["RightAmygdala", "RightHippocampus"]), True)
        self.assertEqual(np.array_equal(XY[2], ["RightHippocampus", "RightAmygdala"]), False)

        self.assertEqual(np.array_equal(XY[4], MERGEDB), False)

        print("getXY with MERGED B tested")


if __name__=='__main__':
    unittest.main()
