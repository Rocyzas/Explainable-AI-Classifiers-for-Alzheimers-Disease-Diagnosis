import nibabel as nib
import numpy as np
import csv
import statistics
import os
import glob




# these are absolute paths on my machine
''' =================ONLY BEFORE TRAINING, TO SAVE ADNI IMAGES AS PROCESSED 2D HIPPOCAMPUS npy ARRAYS================ '''
pathSave ="/media/rokas/HDD/Phase2/Hippocampus2Dmax/"#UNCOMMENT THIS ONE to save
# pathSave ="/media/rokas/HDD/Phase2/Hippocampus2Dmean/"#UNCOMMENT THIS ONE to save
# pathSave ="/media/rokas/HDD/Phase2/Hippocampus2Dstd/"#UNCOMMENT THIS ONE to save

ADNIIMAGES = "/media/rokas/HDD/ADNI/temp/"

pathL = "*_mask_L.nii.gz"
pathR = "*_mask_R.nii.gz"


def Dimentionality_Reduction():
    array_of_2d_images = []

    maxArr = np.zeros([1000, 1000]) # will trim zeros later
    i=0
    arrayOfPathsL=[]
    # [arrayOfPathsL.append(file) for file in sorted(glob.glob("/media/rokas/HDD/testADNICase/progressofReaders/"+pathL))]
    [arrayOfPathsL.append(file) for file in sorted(glob.glob(ADNIIMAGES+pathL))]
    arrayOfPathsR=[]
    # [arrayOfPathsR.append(file) for file in sorted(glob.glob("/media/rokas/HDD/testADNICase/progressofReaders/"+pathR))]
    [arrayOfPathsR.append(file) for file in sorted(glob.glob(ADNIIMAGES+pathR))]

    for file in arrayOfPathsL:

        img = nib.load(file).get_fdata()

        single_2d_image=np.zeros(img[:,:,1].shape)

        for x in range(img[:,:,0][0].shape[0]):
            for y in range(img[:,:,0][0].shape[0]):
                single_2d_image[x][y] = int(np.std(img[x][y]))

        ''' Removing Zero lines and columnss'''
        # removing rows of 0's
        single_2d_image=single_2d_image[~np.all(single_2d_image == 0, axis=1)]

        # removing columns of 0's
        idx = np.argwhere(np.all(single_2d_image[..., :] == 0, axis=0))
        single_2d_image = np.delete(single_2d_image, idx, axis=1)

        # appending this image data to a general images list
        array_of_2d_images.append(single_2d_image)
        i+=1
        print(i)

    i=0
    for file in arrayOfPathsR:
        img = nib.load(file).get_fdata()

        single_2d_image=np.zeros(img[:,:,1].shape)

        # exit()
        for x in range(img[:,:,0][0].shape[0]):
            for y in range(img[:,:,0][0].shape[0]):
                single_2d_image[x][y] = int(np.std(img[x][y]))

        ''' Removing Zero lines and columnss'''
        # removing rows of 0's
        single_2d_image=single_2d_image[~np.all(single_2d_image == 0, axis=1)]

        # removing columns of 0's
        idx = np.argwhere(np.all(single_2d_image[..., :] == 0, axis=0))
        single_2d_image = np.delete(single_2d_image, idx, axis=1)

        # appending this image data to a general images list
        array_of_2d_images.append(single_2d_image)
        i+=1
        print(i)

    # TODO: for USER CASE, save these shapes, so that user input might be appended with 0's
    # x_max = 60
    x_max = max(
            max(array_of_2d_images[i].shape[0] for i in range(len(array_of_2d_images))),
            max(array_of_2d_images[i].shape[0] for i in range(len(array_of_2d_images))))
    # y_max = 60
    y_max = max(array_of_2d_images[i].shape[1] for i in range(len(array_of_2d_images)))

    if x_max%2!=0:x_max+=1
    if y_max%2!=0:y_max+=1
    print("==========================================================")


    halfLength = int(len(array_of_2d_images)/2) #since len array_of_2d_images is always even
    # print("asd", (halfLength))
    m=0
    # for path in [arrayOfPathsL, arrayOfPathsR]:

    ''' SECOND METHOD IS TO SPLIT array_of_2d_images into left and right and then save it along with path right and left'''
    for i in range(halfLength):
        # always restarting array back to 0's
        result = np.zeros([x_max, y_max])
        x_min=array_of_2d_images[i].shape[0]
        y_min=array_of_2d_images[i].shape[1]
        result[int((x_max-x_min)/2):int(x_max-(x_max-x_min)/2), int((y_max-y_min)/2):int(y_max-(y_max-y_min)/2)] = array_of_2d_images[i]

        # getting the last element of a path (the exact name of a file)
        # print((path[i].rsplit('/', 1)[1]).split('.'))
        # print("=============")
        splt_fileName = (arrayOfPathsL[i].rsplit('/', 1)[1]).split('.')[0]
        np.save(pathSave+splt_fileName, result.astype(int))


        result = np.zeros([x_max, y_max])
        x_min=array_of_2d_images[i+halfLength].shape[0]
        y_min=array_of_2d_images[i+halfLength].shape[1]
        result[int((x_max-x_min)/2):int(x_max-(x_max-x_min)/2), int((y_max-y_min)/2):int(y_max-(y_max-y_min)/2)] = array_of_2d_images[i+halfLength]

        splt_fileName = (arrayOfPathsR[i].rsplit('/', 1)[1]).split('.')[0]
        np.save(pathSave+splt_fileName, result.astype(int))

        print(m, " - ", result.shape, "   =", result.mean())

        # np.savetxt(pathSave+splt_fileName+".csv", result.astype(int), fmt='%i', delimiter=",")
        print("DATA READ")


Dimentionality_Reduction()
