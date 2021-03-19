import os
import numpy as np
from nibabel.testing import data_path
filename = 'ADNI_941_S_1363_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070801202916120_S28008_I63897.nii'

# print(data_path)
# exit()
example_filename = os.path.join( filename)

import nibabel as nib
img = nib.load(example_filename)
print(img.shape)
