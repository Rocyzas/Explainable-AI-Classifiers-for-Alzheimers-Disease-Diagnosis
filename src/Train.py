import time
import numpy as np
import random
import sys

from DataProcessing import *
from LRclf import LR
from SVMclf import SVM
from DTclf import DT
from PythonParser import parserTrainARGV

from Explainer import feature_importance

def add(a, b):
    return a+b

def main(argv):

    args = parserTrainARGV(argv)

    # Loading data for any scenario
    df = getDf()

    if args.classification=='ALL':
        classesList = ['HC_AD', 'MCI_AD', 'HC_MCI']
    else:
        classesList = [args.classification]


    # if one is specified, goes only though the loop once
    for classification in classesList:

        XY = getXY(df, classification)
        X, y = shuffleZipData(XY[0], XY[1])

        models = []
        names = []

        if args.classifier=='DT' or args.classifier=='ALL':
            names.append("DT")
            models.append(DT(X, y, int(args.BSCV), classification))

        if args.classifier=='LR' or args.classifier=='ALL':
            names.append("LR")
            models.append(LR(X, y, int(args.BSCV), classification))

        if args.classifier=='SVM' or args.classifier=='ALL':
            names.append("SVM")
            models.append(SVM(X, y, int(args.BSCV), classification))

        if args.save==True:
            saveDataFiles(classification, XY[4])

            for name, model in zip(names, models):
                saveModel(model, name, classification)
                feature_importance(model, name, classification, XY[2])

    if args.save==True:
        saveFilesOnce(df)

if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method', 0/1 for selecting best hyper 0/1 save
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
