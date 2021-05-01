import nibabel as nib
import numpy as np
import csv
import statistics
import os
import glob
import time

# TODO: Mean, standart deviation, max..
# TODO: Fix the paths
# temporary path
# pathSaveL ="/media/rokas/HDD/ADNI/Processed2DHippocampalLeftMaxA/"
# pathSaveR ="/media/rokas/HDD/ADNI/Processed2DHippocampalRightMaxA/"
pathSave ="/media/rokas/HDD/Phase2/Hippocampus3D/"
pathL = "*_mask_L.nii.gz"
pathR = "*_mask_R.nii.gz"


def Dimentionality_Reduction():
    array_of_3d_images = []

    maxArr = np.zeros([1000, 1000]) # will trim zeros later
    i=0
    arrayOfPathsL=[]
    [arrayOfPathsL.append(file) for file in sorted(glob.glob("/media/rokas/HDD/ADNI/temp/"+pathL))]
    arrayOfPathsR=[]
    [arrayOfPathsR.append(file) for file in sorted(glob.glob("/media/rokas/HDD/ADNI/temp/"+pathR))]

    new_x, new_y, new_z = 96, 96, 88
    for file in arrayOfPathsL:
        img = nib.load(file).get_fdata()
        x, y, z = img.shape

        distanceToSubtractXY = int((x-new_x)/2) #same for x and y
        distanceToSubtractZ = int((z-new_z)/2)

        start_time = time.time()
        # subyb=0
        newMat=np.zeros([new_x,new_y,new_z])
        mm = 0
        for m in range(distanceToSubtractXY, new_x+distanceToSubtractXY, 1):
            nn=0
            for n in range(distanceToSubtractXY, new_y+distanceToSubtractXY, 1):
                oo=0
                for o in range(distanceToSubtractZ, new_z+distanceToSubtractZ, 1):
                    newMat[mm,nn,oo] = img[m, n, o]
                    # subyb += img[m, n, o]
                    oo+=1
                nn+=1
            mm+=1

        splt_fileName = (file.rsplit('/', 1)[1]).split('.')[0]
        np.save(pathSave+splt_fileName, newMat.astype(int))
        # array_of_3d_images.append(newMat)

        print(i)
        i+=1

    i=0
    for file in arrayOfPathsR:
        img = nib.load(file).get_fdata()
        x, y, z = img.shape

        distanceToSubtractXY = int((x-new_x)/2) #same for x and y
        distanceToSubtractZ = int((z-new_z)/2)

        start_time = time.time()
        # subyb=0
        newMat=np.zeros([new_x,new_y,new_z])
        mm = 0
        for m in range(distanceToSubtractXY, new_x+distanceToSubtractXY, 1):
            nn=0
            for n in range(distanceToSubtractXY, new_y+distanceToSubtractXY, 1):
                oo=0
                for o in range(distanceToSubtractZ, new_z+distanceToSubtractZ, 1):
                    newMat[mm,nn,oo] = img[m, n, o]
                    # subyb += img[m, n, o]
                    oo+=1
                nn+=1
            mm+=1
        # array_of_3d_images.append(newMat)
        splt_fileName = (file.rsplit('/', 1)[1]).split('.')[0]
        np.save(pathSave+splt_fileName, newMat.astype(int))
        print(i)
        i+=1


    # for i in range(len(array_of_3d_images)):
    #     for path in [arrayOfPathsL, arrayOfPathsR]:
    #         # getting the last element of a path (the exact name of a file)
    #         splt_fileName = (path[i].rsplit('/', 1)[1]).split('.')[0]
    #         # print(m, " - ", result.shape, "   =", result.mean())
    #
    #         # np.savetxt(pathSave+splt_fileName+".csv", result.astype(int), fmt='%i', delimiter=",")
    #         np.save(pathSave+splt_fileName, array_of_3d_images[i].astype(int))


Dimentionality_Reduction()
