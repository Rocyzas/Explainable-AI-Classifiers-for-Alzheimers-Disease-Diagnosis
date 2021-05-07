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

from params import *

def data_processing(TwoD=True, iProjection=None, typeExclude=None):

    pathR = pathHippocampusofProjection + iProjection
    # pathR = '/media/rokas/HDD/ADNI/HippoTest' #60x60
    # pathLabels = "/media/rokas/HDD/ADNI/ADNI1_Screening_1.5T_3_21_2021.csv"

    arrayOfNPY = []
    arraySubject = [] #Names of subject

    arrayLabels = []
    arraySubjectsGroups = []

    with open(pathToADNILabels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) #skipping first line
        seen = []
        for row in csv_reader:
            # not dict, because multiple keys appears

            if row[1] in seen: continue # skip duplicate

            seen.append(row[1])
            arraySubjectsGroups.append(row[1])
            arrayLabels.append(row[2])

    # Getting data from  hippocampal area
    for filePath in sorted(glob.glob(pathR + "/*")):
        splt_fileName = '_'.join(((filePath.rsplit('/', 1)[1]).split('.')[0]).split('_')[1:4])
        arrayOfNPY.append(np.load(filePath))
        arraySubject.append(splt_fileName)

    # Mapping Labels and Data together (indirectly, since key values are repetative, but too importand to drop out)
    X=[] #data
    Y=[] #labels

    for Data in range(len(arraySubject)):
        for Label in range(len(arraySubjectsGroups)):
            if arraySubject[Data] == arraySubjectsGroups[Label]:
                X.append(arrayOfNPY[Data])
                Y.append(arrayLabels[Label])

    # for binary but this changes labels itself


    if typeExclude != "multi":
        X[:] = [X[value] for value in range(len(X)) if Y[value]!=typeExclude]
        Y[:] = [Y[value] for value in range(len(Y)) if Y[value]!=typeExclude]

    # one hot encode This will sort() alphabeticaly
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    x_max = max(X[i].shape[0] for i in range(len(X)))
    y_max = max(X[i].shape[1] for i in range(len(X)))

    X = np.array(X)


    # TODO: should use numpy reshape to be more  efficient
    if TwoD:
        mX = []
        for d in X:
            mX.append(d.flatten())
        X = np.array(mX)

    X, Y = shuffle(X, Y)

    print("Data Processing Finished")

    return X, Y
