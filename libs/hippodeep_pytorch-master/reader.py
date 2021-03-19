import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib

hR = 'RADNI_003_S_0908_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727175013471_S32516_I62589_mask_R.nii.gz'
hL = 'LDNI_003_S_0908_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727175013471_S32516_I62589_mask_L.nii.gz'
# eB = 'example_brain_t1.nii.gz'
oA = 'ORGADNI_003_S_0908_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727175013471_S32516_I62589.nii'

hr = nib.load(hR)
hl = nib.load(hL)
# eb = nib.load(eB)
oa = nib.load(oA)
sx, sy, sz = hr.header.get_zooms()
volume = sx * sy * sz
print(sx, sy, sz)

sx, sy, sz = hl.header.get_zooms()
volume = sx * sy * sz
print(sx, sy, sz)

sx, sy, sz = oa.header.get_zooms()
volume = sx * sy * sz
print(sx, sy, sz)
