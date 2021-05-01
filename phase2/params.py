
# RESOURCES
IMAGESPATH = "/media/rokas/HDD/Phase2/testUserImage/*" #CAREFUL, since it will remove REMOVEFILES from the folder
MODELSLOCATION = "/media/rokas/HDD/Phase2/FinalModels/Models/"
PATHHIPPODEEP = "/media/rokas/HDD/hippodeep_pytorch-master"

# SAVING
PATHSAVE = "/media/rokas/HDD/Phase2/testUserImage/"
MODELSLOG = "/media/rokas/HDD/Phase2/testUserImage/predictedclassesLog.csv"

# REMOVING
REMOVEFILES = "rm *.gz *.npy *.warning.txt"

# SELECT CLASSIFICATION(multi, AD, MCI, CN) METHOD
# AD - removes AD cases, keeping MCI vs. CN
# MCI - removes MCI cases, keeping AD vs. CN
# CN - removes CN cases, keeping AD vs. MCI
CLASSIFICATION = "multi"

# SELECT INTENSITY PROJECTION(mean, std, max)
PROJECTIONS = ["mean", "std", "max"]
