
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

Xval = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 113, 0, 15])
XvalZ = np.zeros(len(Xval))

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
    # 
    # def test_2_shuffleZipData(self):
    #     assert not np.array_equal(DataProcessing.shuffleZipData(Xval)[0], Xval)
    #     assert not np.array_equal(DataProcessing.shuffleZipData(Xval, XvalZ)[0], Xval)
    #     np.array_equal(DataProcessing.shuffleZipData(Xval, XvalZ)[1], XvalZ)
    #     np.array_equal(DataProcessing.shuffleZipData(XvalZ)[0], XvalZ)
    #
    #     print("Shuffle Zip Data Tested")

    def test_3_scaleData(self):
        # print("BV: ", Xval.reshape(-1, 1))
        scaledData = DataProcessing.scaleData(Xval.reshape(-1, 1))
        # print("BV: ", scaledData)
        # print("THE sc ", scaledData)
        backData = DataProcessing.scaleData(scaledData)
        # print("THE BAC: ",backData)
        #
        # print("Scale Function Tested")


if __name__=='__main__':
    unittest.main()
