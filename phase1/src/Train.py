import numpy as np
import random
import sys

from DataProcessing import getXY, getDf
from PythonParser import parserTrainARGV
from save_load import saveDataFiles, saveModel, feature_importance, saveFilesOnce

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from classificationModels import createModels

def trainEachClassifier(argv):
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
        X, y = shuffle(XY[0], XY[1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        obj = createModels(X_train, X_test, y_train, y_test, classification)

        models = []
        names = []

        if args.classifier=='DT' or args.classifier=='ALL':
            names.append("DT")
            models.append(obj.DT())

        if args.classifier=='LR' or args.classifier=='ALL':
            names.append("LR")
            models.append(obj.LR())

        if args.classifier=='SVM' or args.classifier=='ALL':
            names.append("SVM")
            models.append(obj.SVC())

        if args.save==True:
            saveDataFiles(classification, XY[4])

            for name, model in zip(names, models):
                saveModel(model, name, classification)
                feature_importance(model, name, classification, XY[2])

    if args.save==True:
        saveFilesOnce(df, XY[2])
