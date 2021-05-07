
''' ===================FOR PREDICTION=================== '''
# RESOURCES
IMAGESPATH = "../testUserImage/*" #CAREFUL, since it will remove REMOVEFILES from the folder
MODELSLOCATION = "../FinalModels/Models/"
PATHHIPPODEEP = "../hippodeep_pytorch-master"

# SAVING
PATHSAVE = "../testUserImage/"
MODELSLOG = "../testUserImage/detailedFinalPredictionLog.csv"
FINALPRED = "../testUserImage/finalPredictions.csv"

# REMOVING
REMOVEFILES = "rm *_brain_mask.nii.gz *_cerebrum_mask.nii.gz *.npy *.warning.txt *_hippoLR_volumes.csv *_mask_*.nii.gz"

# SELECT CLASSIFICATION(multi, AD, MCI, CN) METHOD
# AD - removes AD cases, keeping MCI vs. CN
# MCI - removes MCI cases, keeping AD vs. CN
# CN - removes CN cases, keeping AD vs. MCI
CLASSIFICATION = ["multi", "AD", "MCI"]

# SELECT INTENSITY PROJECTION(mean, std, max)
PROJECTIONS = ["mean", "std", "max"]


''' ===================FOR TRAINING=================== '''
# Keras NN
BATCH_SIZE = 32
EPOCHS = 500

# sklearn MLPC
ITERS_MLPC = 3
N_EPOCHS_MLPC=100


FILETOSAVESUMMARY_LENET = '/media/rokas/HDD/Phase2/model.png'

saveModelPath = "../Models/"
modelsLog = "../Metrics/log.csv"
pathForSavingPlots = '../Metrics/'
pathHippocampusofProjection = '../Hippocampus2D'
pathToADNILabels = "../ADNI1_Screening_1.5T_2021.csv"
