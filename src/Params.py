from joblib import load, dump
import os



# ========== Data Processing ============
PATH_sMRI = os.getcwd() +  "/../DataFiles/SheffieldProspective_sMRI.csv"
PATH_ASL = os.getcwd() +  "/../DataFiles/SheffieldProspective_ASL.csv"
PATH_Demo = os.getcwd() +  "/../DataFiles/SheffieldProspective_Demo.csv"
PATH_Neuro = os.getcwd() +  "/../DataFiles/SheffieldProspective_Neuro.csv"

saveColTemplate = "../DataFiles/Columns.csv"
fulldata = "../DataFiles/"
# List the features that should be removed
# listFeaturesRemove = ['ID', 'MCI', 'AD']
listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude', 'CDR']



# ============= Classifiers =============
N_ITER = 5
CV = 5
modelsPath = '/home/rokas/Documents/Leeds/3Year1Term_Works/IndividualProject/Source/Models/'
repetativeScore = 0

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
