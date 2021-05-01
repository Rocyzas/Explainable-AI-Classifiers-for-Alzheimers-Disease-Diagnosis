import numpy as np
import glob
import os
import csv
from keras.utils import np_utils

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def data_processing(TwoD=True):

    # pathL = '/media/rokas/HDD/ADNI/Processed2DHippocampalLeftMax'
    # pathR = '/media/rokas/HDD/ADNI/Processed2DHippocampalRightMax'
    pathR = '/media/rokas/HDD/Phase2/Hippocampus3D'
    pathLabels = "/media/rokas/HDD/ADNI/ADNI1_Screening_1.5T_3_21_2021.csv"

    # arrayOfNPY_Left = []
    # arraySubjectLeft = [] #Names of subject

    arrayOfNPY_Right = []
    arraySubjectRight = [] #Names of subject

    arrayLabels = []
    arraySubjectsGroups = []

    with open(pathLabels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) #skipping first line
        seen = []
        for row in csv_reader:
            # not dict, because multiple keys appears

            if row[1] in seen: continue # skip duplicate

            seen.append(row[1])
            arraySubjectsGroups.append(row[1])
            arrayLabels.append(row[2])

    # Getting data from Left hippocampal area
    # for filePath in sorted(glob.glob(pathL + "/*")):
    #     splt_fileName = '_'.join(((filePath.rsplit('/', 1)[1]).split('.')[0]).split('_')[1:4])
    #     arrayOfNPY_Left.append(np.load(filePath))
    #     arraySubjectLeft.append(splt_fileName)

    # Getting data from Right hippocampal area
    i=0
    for filePath in sorted(glob.glob(pathR + "/*")):
        splt_fileName = '_'.join(((filePath.rsplit('/', 1)[1]).split('.')[0]).split('_')[1:4])
        arrayOfNPY_Right.append(np.load(filePath))
        arraySubjectRight.append(splt_fileName)
        print(i)
        i+=1

    # print(len(arrayOfNPY_Left), len(arrayOfNPY_Right), len(arraySubjectLeft), len(arraySubjectRight))

    # Mapping Labels and Data together (indirectly, since key values are repetative, but too importand to drop out)
    X=[] #data
    Y=[] #labels

    # for Data in range(len(arraySubjectLeft)):
    #     # Getting data all together (left and right hippos)
    #     '''LEFT'''
    #     for Label in range(len(arraySubjectsGroups)):
    #         if arraySubjectLeft[Data] == arraySubjectsGroups[Label]:
    #             X.append(arrayOfNPY_Left[Data])
    #             Y.append(arrayLabels[Label])
    #             # print(m, " - ", X[Data].shape, "   =", X[Data].mean())

    '''RIGHT'''
    for Data in range(len(arraySubjectRight)):
        for Label in range(len(arraySubjectsGroups)):
            if arraySubjectRight[Data] == arraySubjectsGroups[Label]:
                X.append(arrayOfNPY_Right[Data])
                Y.append(arrayLabels[Label])

    # one hot encode
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)


    x_max = max(X[i].shape[0] for i in range(len(X)))
    y_max = max(X[i].shape[1] for i in range(len(X)))

    X = np.array(X)
    # print(X.shape, type(X))
    # print(Y.shape, type(Y))
    # print(Y)
    # Y1 = []
    # for y in Y:
    #     Y1.append([y])
    # Y=np.array(Y1)
    # print(Y)
    # print(Y.shape, type(Y))
    # # exit()
    # X = np.array(X)
    # print(type(X), X.shape)
    # X = np.expand_dims((X), axis=-1)

    # TODO: should use numpy reshape to be more  efficient
    if TwoD:
        mX = []
        for d in X:
            mX.append(d.flatten())
        X = np.array(mX)
    # print(Y.shape)
    # #
    # Y=Y.ravel()
    # print(Y.shape)
    X, Y = shuffle(X, Y)


    print("Data Processing Finished")
    # exit()
    return X, Y
