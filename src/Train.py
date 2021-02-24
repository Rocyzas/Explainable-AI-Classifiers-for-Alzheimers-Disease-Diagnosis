import time
import numpy as np
import statistics
import random
import sys
from math import sqrt

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DataProcessing import *
# from Params import *
from LRclf import LR
from SVMclf import SVM
from DTclf import DT

def main(argv):

    # Loading data for any scenario
    df = getDf()

    if argv[1]=='ALL':
        classesList = ['HC_AD', 'MCI_AD']
    else:
        classesList = [argv[1]]


    for classificationType in classesList:
        XY = getXY(df, classificationType)

        X, y = shuffleZipData(XY[0], XY[1])


        models = []
        names = []

        if argv[0]=='DT' or argv[0]=='ALL':
            names.append("DT")
            models.append(DT(X, y, int(argv[2]), classificationType))

        if argv[0]=='SVM' or argv[0]=='ALL':
            names.append("SVM")
            models.append(SVM(X, y, int(argv[2]), classificationType))

        if argv[0]=='LR' or argv[0]=='ALL':
            names.append("LR")
            models.append(LR(X, y, int(argv[2]), classificationType))

        elif argv[0]!='LR' and argv[0]!='DT' and argv[0]!='SVM' and argv[0]!='ALL':
            exit(0)

        if argv[3]=='1':
            # clf, clfName, classes
            for name, model in zip(names, models):
                saveModel(model, name, classificationType)

if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method' '0/1(explainability)', 0/1 for selecting best hyper
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
