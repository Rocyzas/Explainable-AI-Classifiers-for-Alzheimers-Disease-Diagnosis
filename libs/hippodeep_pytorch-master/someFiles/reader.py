import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib

# orig_ADNI = 'ORGADNI_003_S_0908_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727175013471_S32516_I62589.nii'
# R_adni = 'RADNI_003_S_0908_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727175013471_S32516_I62589_mask_R.nii.gz'
# x='ADNI_003_S_0908_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070727175013471_S32516_I62589_cerebrum_mask.nii.gz'
# e='example_brain_t1.nii.gz'
# newo='ADNI_002_S_0729_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217001935821_S16874_I40708.nii'


# R_filename = os.path.join(R_adni)
# img = nib.load(R_filename)
# print(img.shape)
#
# N = os.path.join(newo)
# imgN = nib.load(N)
# print(imgN.shape)
#
# data = img.get_fdata()
# # data_2d = data.reshape(data, 2)
# # data = img.get_data()
# print("MAMAMIJA")
# print(data.shape)
# # print(data_2d.shape)

AR=['A_mask_R.nii.gz', 'A_mask_L.nii.gz',
'B_mask_R.nii.gz',
'B_mask_L.nii.gz',
'C_mask_R.nii.gz',
'C_mask_L.nii.gz',
'D_mask_R.nii.gz',
'D_mask_L.nii.gz',
'd_mask_R.nii.gz',
'E_mask_L.nii.gz',
'F_mask_L.nii.gz',
'G_mask_L.nii.gz',
'H_mask_L.nii.gz',
'I_mask_L.nii.gz',

'E_mask_R.nii.gz',
'F_mask_R.nii.gz',
'G_mask_R.nii.gz',
'H_mask_R.nii.gz',
'I_mask_R.nii.gz']

for file in AR:
    img = nib.load(os.path.join(file))
    print(img.shape)
# AL='A_mask_L.nii.gz'
#
# BR='B_mask_R.nii.gz'
# BL='B_mask_L.nii.gz'
#
# CR='C_mask_R.nii.gz'
# CL='C_mask_L.nii.gz'
#
# DR='D_mask_R.nii.gz'
# DL='D_mask_L.nii.gz'
#
# dR='d_mask_R.nii.gz'
# dL='d_mask_L.nii.gz'

# R_filename = os.path.join(R_adni)
# img = nib.load(R_filename)
# print(img.shape)
