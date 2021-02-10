import time
import numpy as np
import statistics
import random
import sys
from math import sqrt

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DataProcessing import *
from Classifiers import *

def loadModel(name):
    model = load('Models/' + name + '.joblib')
    return model


def main(argv):

    # Getting XY for training and testint !=DT since it does not require scaling
    XY = getXY(argv[0]!='DT', argv[1])

    if argv[0]=='DT':
        model = DT(XY[0], XY[1], int(argv[2]))

    elif argv[0]=='SVM':
        model = SVM(XY[0], XY[1], int(argv[2]))

    elif argv[0]=='LR':
        model = LR(XY[0], XY[1], int(argv[2]))

    elif argv[0]=='best':
        model = selectBestClassifier()

    else:
        exit(0)

    if argv[3]=='1':
        # clf, clfName, classes
        saveModel(model, argv[0], argv[1])

if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method' '0/1(explainability)', 0/1 for selecting best hyper
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
