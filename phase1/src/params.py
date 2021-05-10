import os
from skopt.space import Real, Categorical, Integer
# ========== Data Processing ============
# Sheffiled data sets
PATH_sMRI_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_sMRI.csv"
PATH_ASL_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_ASL.csv"
PATH_Demo_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_Demo.csv"
PATH_Neuro_Sheffield = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/SheffieldProspective_Neuro.csv"

# Combined Files (Sheffield, ADNI, hippodeep)
PATH_synthetic = os.getcwd() +  "/../DataFiles/SyntheticData.csv"
PATH_grouped = os.getcwd() +  "/../DataFiles/grouped.csv"
PATH_FINAL = os.getcwd() +  "/../DataFiles/FINAL.csv"
PATH_F01 = os.getcwd() +  "/../DataFiles/F01.csv"

# ADNI data sets
PATH_sMRI_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_sMRI.csv"
PATH_ASL_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_ASL.csv"
PATH_Demo_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_Demo.csv"
PATH_Neuro_ADNI = os.getcwd() +  "/../DataFiles/VPH-DARE_EXTENDED/ADNI_Neuro.csv"


saveColTemplate = "../DataFiles/Columns.csv"
fulldata = "../DataFiles/"
predictionFile = "Predictions.csv"

# "ADNI", "SHEFF", "BOTH"
whichDatasets="ADNI"

# List the features that should be removed
# listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude', 'Age', 'Gender', 'Height', 'Weight']
listFeaturesRemove = ['ID', 'MCI', 'AD', 'Exclude', 'Age','Gender', 'Height', 'Weight']


# normalised brain volume value (ADNI)
valueOfBrainVolumeADNI = 1214750.869

# ============= Classifiers =============
N_ITER = 35
CV = 5

# balanced_accu
'''['accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'recall', 'recall_macro', 'recall_micro',
'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted']
'''

trainingTimesRange = 3
scoringMetrics = 'f1_weighted'
# scoringMetrics = 'accuracy' #accuracy works for both binary and multi

modelsPath = '../Models/FinalModels_4features/'
# modelsPath = '../Models/'

repetativeScore = 0

modelsLog = "../Models/log.csv"

ExplainPath = '../ExplainHTML/'

search_spaceDT = {
    "max_depth": Integer(5, 20),
    "max_features": Categorical(['auto', 'sqrt','log2']),
    "min_samples_leaf": Integer(1, 10),
    "min_samples_split": Real(0.001, 1),
    "n_estimators": Integer(50, 500),
    'learning_rate':Real(0.000001, 1)
}
search_spaceSVC = {
    "kernel":Categorical(['poly', 'rbf', 'sigmoid', 'linear']), #eli5 works only with linear
    "degree":Integer(0, 10),
    "gamma":Real(1e-5, 1e-1),
    "coef0":Real(0, 1),
    "tol":Real(1e-5, 1e-1),
    "C":Real(1e-6, 1),
    "class_weight":Categorical(['balanced'])
}

search_spaceLR = {
    "penalty":Categorical(['l2']),
    "solver":Categorical(['lbfgs','saga']),
    "multi_class":Categorical(['multinomial']),
    "tol":Real(1e-5, 1e-1),
    "C":Real(0.1, 10),
    "class_weight":Categorical(['balanced'])
}


filterSpecificFeatures = [
# 'Brainstem.1',
# 'Left Cortex',
# 'Right Cortex',
'LeftAmygdala',
'RightAmygdala',
'LeftHippocampus',
'RightHippocampus',
# '3rdVentricle',
# 'RightLateralVentricle',
# 'LeftCerebralWhiteMatter',
# 'RightCerebralWhiteMatter'
]
