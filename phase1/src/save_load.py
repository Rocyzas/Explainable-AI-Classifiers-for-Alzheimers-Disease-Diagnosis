from joblib import load, dump
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from params import *

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def loadModel(name):
    model = load(modelsPath + name + '.joblib')
    return model


def saveModel(model, modelName, classes):
    name = modelName+"-"+classes
    dump(model, uniquify(modelsPath + name +'.joblib'))
    return


def saveDataFiles(classes, dfProcessed):
    # processed df (without labels, only training, unscaled data)
    dfProcessed.to_csv(fulldata + classes + '_DataUsed.csv', sep=',')
    print("Cleaned and Processed dataframe saved")
    return


def saveFilesOnce(df, cols):
    # cleaned data with with all columns and labels
    df.to_csv(fulldata + 'FullData.csv', sep=',')
    df.filter(cols).to_csv(fulldata + 'UsedData.csv', sep=',')

    with open(saveColTemplate, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(cols)
        f.close()
    print("Full dataframe csv file saved")
    return


def logClassifier(clf, classes, accuracy, recall, precision, f1, roc_auc, report, matrix, parameters, scoring):

    import datetime
    utc_datetime = datetime.datetime.utcnow()
    time = utc_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # If log does not exist, create Column names
    if os.path.exists(modelsLog)!=True:
        with open(modelsLog, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Classifier", "Classification Groups",
                            "Accuracy", "Recall", "Precision", "F-score",
                            "ROC_AUC", "Report", "Confusion Matrix", "HyperParameters", "Scoring Metrics", "Time"])
            f.close()

    # append existing 'modelsLog' file
    with open(modelsLog, mode='a+') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([clf.__class__.__name__, classes, accuracy, recall, precision, f1,
                        roc_auc, report, matrix, parameters, scoring, time])

    f.close()

    return

def save_predictions(i, diagnosis, classes, clf):

    with open(predictionFile, mode='a+') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([i, diagnosis, classes, clf.__class__.__name__])
        f.close()
    print("Predictions Saved")
    return


def feature_importance(clf, name, classes, columns):

        numberOfImportatnFeatures_toShow = 10
        cols = list(columns)
        fc=[]

        if clf.__class__.__name__=="GradientBoostingClassifier":
            print("Feature importance on ", clf.__class__.__name__)

            importances = clf.feature_importances_

            std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

            indices = np.argsort(importances)[::-1]

            for i in indices:
                fc.append(cols[i])


        elif clf.__class__.__name__=="LogisticRegression":
            importances = clf.coef_[0]
            abs_weights = np.abs(importances)

            # indices = np.fliplr(np.argsort(abs_weights))
            indices = (-abs_weights).argsort()[:len(abs_weights)]

            for i in indices:
                fc.append(cols[i])

        # not possible for non-linear kernel (might check why for the report)
        elif clf.__class__.__name__=="SVC" and clf.kernel=="linear":

            importances = clf.coef_[0]
            abs_weights = np.abs(importances)

            indices = (-abs_weights).argsort()[:len(abs_weights)]

            for i in indices:
                fc.append(cols[i])

        else:
            print("Classifier required: GradientBoostingClassifier or LogisticRegression or SVC(linear)")
            return

        plt.figure(figsize=(6,8))
        plt.title("Feature importances " + str(clf.__class__.__name__))
        plt.bar(range(len(columns)), importances[indices], color="r", align="center")
        plt.xticks(range(len(columns)), fc, rotation='vertical')
        plt.subplots_adjust(bottom=0.25)
        plt.xlim([-0.5, numberOfImportatnFeatures_toShow])


        try:
            path = modelsPath + "FeatureImportance"
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed or already exist" % path)

        path = path + "/" + classes
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed or already exist" % path)

        plt.savefig(uniquify(path  + "/" + name + "_Feature_Importance.png"))

        plt.close('all')
