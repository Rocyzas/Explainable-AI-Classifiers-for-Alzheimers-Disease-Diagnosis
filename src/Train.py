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

    # Getting XY for training and testint !=DT since it does not require scaling
    # XY = getXY(argv[0]!='DT', argv[1])
    XY = getXY(argv[1])

    XYcombination = list(zip(XY[0], XY[1]))
    random.shuffle(XYcombination)
    X, y = zip(*XYcombination)

    models = []

    if argv[0]=='DT' or argv[0]=='ALL':
        models.append(DT(X, y, int(argv[2])))

    if argv[0]=='SVM' or argv[0]=='ALL':
        models.append(SVM(X, y, int(argv[2])))

    if argv[0]=='LR' or argv[0]=='ALL':
        models.append(LR(X, y, int(argv[2])))
    #
    # elif argv[0]=='all':
    #     models.append(DT(X, y, int(argv[2])))
    #     models.append(SVM(X, y, int(argv[2])))
    #     models.append(LR(X, y, int(argv[2])))
    #     # Should select from the directory of ALL models
    #     # but makes not sense if data is different.
    #     # model = selectBestClassifier()

    else:
        exit(0)

    if argv[3]=='1':
        # clf, clfName, classes
        for model in models:
            saveModel(model, argv[0], argv[1])

if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method' '0/1(explainability)', 0/1 for selecting best hyper
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
