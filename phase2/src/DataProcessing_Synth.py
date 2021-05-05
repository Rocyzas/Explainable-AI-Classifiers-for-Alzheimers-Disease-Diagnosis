import numpy as np
import glob
import os
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def data_processing(TwoD=True):

    pathR = '/media/rokas/HDD/Phase2/SyntheticData'

    arrayOfNPY=[]

    for filePath in sorted(glob.glob(pathR + "/*.csv")):
        # print(filePath)
        arrayOfNPY.append(np.genfromtxt(filePath, delimiter=','))

    '''RIGHT'''
    X=arrayOfNPY
    Y=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #correct  labels
    # Y=[0,1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1] #random labels

    # one hot encode
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    X = np.array(X)

    if TwoD:
        mX = []
        for d in X:
            mX.append(d.flatten())
        X = np.array(mX)

    X, Y = shuffle(X, Y)

    print("Data Processing Finished")

    return X, Y

# data_processing()
