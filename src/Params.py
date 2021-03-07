from joblib import load, dump
import os
import csv
import pandas as pd

# ========== Data Processing ============
# Sheffiled data sets
PATH_sMRI_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_sMRI.csv"
PATH_ASL_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_ASL.csv"
PATH_Demo_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_Demo.csv"
PATH_Neuro_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_Neuro.csv"
PATH_synthetic = os.getcwd() +  "/../DataFiles/SyntheticData.csv"
# ADNI data sets
PATH_sMRI_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_sMRI.csv"
PATH_ASL_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_ASL.csv"
PATH_Demo_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_Demo.csv"
PATH_Neuro_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_Neuro.csv"


saveColTemplate = "../DataFiles/Columns.csv"
fulldata = "../DataFiles/"
# List the features that should be removed
listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude']
# listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude', 'CDR']

# normalised brain volume value (ADNI)
valueOfBrainVolumeADNI = 1214750.869


# ============= Classifiers =============
N_ITER = 12
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
    name = modelName+"-"+classes
    dump(model, uniquify(modelsPath + name +'.joblib'))

def saveDataFiles(classes, dfProcessed):
    # processed df (without labels, only training, unscaled data)
    dfProcessed.to_csv(fulldata + classes + '_DataUsed.csv', sep=',')
    print("Cleaned and Processed dataframe saved")

def saveFilesOnce(df, columns):

    # cleaned data with with all columns and labels
    df.to_csv(fulldata + 'FullData.csv', sep=',')

    # columns that are used in training(except labels)
    # user should use this 'template' for explanation
    columns = pd.DataFrame(columns = columns)
    columns.to_csv(saveColTemplate, sep=',', index=False)
    print("Full dataframe and columns csv file saved")

def logClassifier(clf, classes, score, confMatrix,  parameters, y, y_pred):
    from sklearn.metrics import roc_auc_score, recall_score
    import datetime
    utc_datetime = datetime.datetime.utcnow()
    time = utc_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # If log does not exist, create Column names
    if os.path.exists(modelsLog)!=True:
        with open(modelsLog, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Classifier", "Classification Between", "Best Accuracy",
                "Confusion Matrix", "Parameters", "ROC_AUC score","Recall", "Time"])
            f.close()

    # append existing 'modelsLog' file
    with open(modelsLog, mode='a+') as file_object:
        writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([clf, classes, score, confMatrix,  parameters,
        roc_auc_score(y, y_pred), recall_score(y, y_pred), time])

    file_object.close()

    return
