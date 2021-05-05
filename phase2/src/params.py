
# RESOURCES
IMAGESPATH = "../testUserImage/*" #CAREFUL, since it will remove REMOVEFILES from the folder
MODELSLOCATION = "../FinalModels/Models/"
PATHHIPPODEEP = "../hippodeep_pytorch-master"

# SAVING
PATHSAVE = "../testUserImage/"
MODELSLOG = "../testUserImage/predictedclassesLog.csv"

# REMOVING
REMOVEFILES = "rm *.gz *.npy *.warning.txt"

# SELECT CLASSIFICATION(multi, AD, MCI, CN) METHOD
# AD - removes AD cases, keeping MCI vs. CN
# MCI - removes MCI cases, keeping AD vs. CN
# CN - removes CN cases, keeping AD vs. MCI
CLASSIFICATION = "multi"

# SELECT INTENSITY PROJECTION(mean, std, max)
PROJECTIONS = ["mean", "std", "max"]

# Keras NN
BATCH_SIZE = 32
EPOCHS = 5

# sklearn MLPC
ITERS_MLPC = 1
N_EPOCHS_MLPC=100


# saveModelPath = "/media/rokas/HDD/Phase2/Models/"
# modelsLog = "/media/rokas/HDD/Phase2/Metrics/logTEMP3.csv"
# pathForSavingPlots = '/media/rokas/HDD/Phase2/Metrics/'
# pathRightHippocampus = '/media/rokas/HDD/Phase2/Hippocampus2D'

saveModelPath = "../Models/"
modelsLog = "../Metrics/log.csv"
pathForSavingPlots = '../Metrics/'
pathHippocampus = '../Hippocampus2D'
