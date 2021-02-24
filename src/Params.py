from joblib import load, dump
import os
import csv


visiskaiGlobalus = 0

# ========== Data Processing ============
# Sheffiled data sets
# PATH_sMRI = os.getcwd() +  "/../DataFiles/SheffieldProspective_sMRI.csv"
# PATH_ASL = os.getcwd() +  "/../DataFiles/SheffieldProspective_ASL.csv"
# PATH_Demo = os.getcwd() +  "/../DataFiles/SheffieldProspective_Demo.csv"
# PATH_Neuro = os.getcwd() +  "/../DataFiles/SheffieldProspective_Neuro.csv"
#
# ADNI data sets
PATH_sMRI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_sMRI.csv"
PATH_ASL = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_ASL.csv"
PATH_Demo = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_Demo.csv"
PATH_Neuro = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_Neuro.csv"


saveColTemplate = "../DataFiles/Columns.csv"
fulldata = "../DataFiles/"
# List the features that should be removed
listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude']
# listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude', 'CDR']



# ============= Classifiers =============
N_ITER = 5
CV = 5
modelsPath = '../Models/'
repetativeScore = 0

modelsLog = "../Models/log.csv"

ExplainPath = '../ExplainHTML/'

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
    # clf = load('Models/' + filename.joblib')
    name = modelName+"-"+classes
    dump(model, uniquify(modelsPath + name +'.joblib'))

def logClassifier(clf, classes, score, confMatrix,  parameters):

    if os.path.exists(modelsLog)!=True:
        with open(modelsLog, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Classifier", "Classification Between", "Best Accuracy", "Confusion Matrix", "Parameters"])
            f.close()

    with open(modelsLog, mode='a+') as file_object:
        writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow([clf, classes, score, confMatrix,  parameters])

    file_object.close()

    return
