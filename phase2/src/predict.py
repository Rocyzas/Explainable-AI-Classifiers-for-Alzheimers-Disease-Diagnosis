import time
import os
import csv
import glob
import numpy as np
import nibabel as nib
from tensorflow import keras

from params import *

def hippodeepToTwoHippos():
    os.chdir(PATHHIPPODEEP)
    os.system("./deepseg1.sh " + IMAGESPATH)
    return

def convertTo2D(projection):
    os.chdir('/'.join(IMAGESPATH.split('/')[:-1]))
    pathL = "*_mask_L.nii.gz"
    pathR = "*_mask_R.nii.gz"
    array_of_2d_images = []

    arrayOfPathsL=[]
    [arrayOfPathsL.append(file) for file in sorted(glob.glob(IMAGESPATH[:-1]+pathL))]
    arrayOfPathsR=[]
    [arrayOfPathsR.append(file) for file in sorted(glob.glob(IMAGESPATH[:-1]+pathR))]
    maxArr = np.zeros([1000, 1000]) # will trim zeros later

    i=0
    for file in arrayOfPathsL:
        img = nib.load(file).get_fdata()

        single_2d_image=np.zeros(img[:,:,1].shape)

        # exit()
        for x in range(img[:,:,0][0].shape[0]):
            for y in range(img[:,:,0][0].shape[0]):
                single_2d_image[x][y] = int(eval("np."+projection)(img[x][y]))

        ''' Removing Zero lines and columnss'''
        # removing rows of 0's
        single_2d_image=single_2d_image[~np.all(single_2d_image == 0, axis=1)]

        # removing columns of 0's
        idx = np.argwhere(np.all(single_2d_image[..., :] == 0, axis=0))
        single_2d_image = np.delete(single_2d_image, idx, axis=1)

        # appending this image data to a general images list
        array_of_2d_images.append(single_2d_image)
        i+=1
        print(i)

    i=0

    for file in arrayOfPathsR:
        img = nib.load(file).get_fdata()

        single_2d_image=np.zeros(img[:,:,1].shape)

        for x in range(img[:,:,0][0].shape[0]):
            for y in range(img[:,:,0][0].shape[0]):
                single_2d_image[x][y] = int(eval("np."+projection)(img[x][y]))

        ''' Removing Zero lines and columnss'''
        # removing rows of 0's
        single_2d_image=single_2d_image[~np.all(single_2d_image == 0, axis=1)]

        # removing columns of 0's
        idx = np.argwhere(np.all(single_2d_image[..., :] == 0, axis=0))
        single_2d_image = np.delete(single_2d_image, idx, axis=1)

        # appending this image data to a general images list
        array_of_2d_images.append(single_2d_image)
        i+=1
        print(i)

    halfLength = int(len(array_of_2d_images)/2) #since len array_of_2d_images is always even

    for i in range(halfLength):
        # always restarting array back to 0's
        result = np.zeros([XMAX_CONST, YMAX_CONST])
        x_min=array_of_2d_images[i].shape[0]
        y_min=array_of_2d_images[i].shape[1]
        result[int((XMAX_CONST-x_min)/2):int(XMAX_CONST-(XMAX_CONST-x_min)/2), int((YMAX_CONST-y_min)/2):int(YMAX_CONST-(YMAX_CONST-y_min)/2)] = array_of_2d_images[i]

        # getting the last element of a path (the exact name of a file)
        splt_fileName = (arrayOfPathsL[i].rsplit('/', 1)[1]).split('.')[0]
        np.save(PATHSAVE+splt_fileName, result.astype(int))


        result = np.zeros([XMAX_CONST, YMAX_CONST])
        x_min=array_of_2d_images[i+halfLength].shape[0]
        y_min=array_of_2d_images[i+halfLength].shape[1]
        result[int((XMAX_CONST-x_min)/2):int(XMAX_CONST-(XMAX_CONST-x_min)/2), int((YMAX_CONST-y_min)/2):int(YMAX_CONST-(YMAX_CONST-y_min)/2)] = array_of_2d_images[i+halfLength]

        splt_fileName = (arrayOfPathsR[i].rsplit('/', 1)[1]).split('.')[0]
        np.save(PATHSAVE+splt_fileName, result.astype(int))

        print(projection, "-->", result.shape, "   =", result.mean())

    return

def decode(yn, classes):
    y=[]

    for i in range(len(yn)):
        if classes=="CN":
            if yn[i]==1:
                prediction = "Mild Cognitive Impairment"
            else:
                prediction = "Alzheimer's Disease"
        elif classes=="MCI":
            if yn[i]==1:
                prediction = "Healthy Case"
            else:
                prediction = "Alzheimer's Disease"
        elif classes=="AD":
            if yn[i]==0:#if CN
                prediction = "Healthy Case"
            else:
                prediction = "Mild Cognitive Impairment"
        elif classes=="multi":
            if yn[i]==1:
                prediction = "Healthy Case"
            elif yn[i]==2:
                prediction = "Mild Cognitive Impairment"
            else:
                prediction = "Alzheimer's Disease"
        y.append(prediction)
        # print("PREDICTION ", i, "-", prediction)

    return y

def predictMLPC(X, arg, projection):

    X = np.array(X)

    mX = []
    for d in X:
        mX.append(d.flatten())
    X = np.array(mX)

    from joblib import load
    clf = load(MODELSLOCATION+"MLPC_"+arg + "_"+projection+".joblib")

    y=[]
    for case in X:
        case = case.reshape(1, -1)
        y.append(clf.predict(case))

    return decode(y, arg)

def predictLeNet(X, arg, projection):
    model = keras.models.load_model(MODELSLOCATION+"LeNet_"+arg + "_"+projection)

    y=[]
    for case in X:
        case = case[np.newaxis, ..., np.newaxis]
        y.append(model.predict_classes(case))

    return decode(y, arg)

def loadImgs():
    # path = os.chdir('/'.join(IMAGESPATH.split('/')[:-1]))
    path = IMAGESPATH.rsplit('/', 1)[0]

    arrayOfNPY = []
    arraySubject = [] #Names of subject

    for filePath in sorted(glob.glob(path + "/*.npy")):
        ''' wond work with different file names so either remove at all or change'''
        # splt_fileName = '_'.join(((filePath.rsplit('/', 1)[1]).split('.')[0]).split('_')[1:4])

        arrayOfNPY.append(np.load(filePath))
        arraySubject.append(filePath)

    return arrayOfNPY, arraySubject

def saveResults(y, name, clf, proj, group):
    # how many cases to be explained

    if os.path.exists(MODELSLOG)!=True:
        with open(MODELSLOG, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Subject", "Diagnosis", "Classifier", "Projection", "Excluded_Group"])
            f.close()

    # append existing 'MODELSLOG' file
    with open(MODELSLOG, mode='a+') as file_object:
        writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(y)):
            writer.writerow([name[i], y[i], clf, proj, group])

    file_object.close()

    return


def removingAllExeptPredictingfiles(val = False):
    os.chdir(IMAGESPATH[:-1])

    os.system(REMOVEFILES)
    if val==True:
        os.system(REMOVEFILES+" "+MODELSLOG.split("/")[-1])
    # os.system("ls -l")
    print(os.getcwd())


def generalise():
    import pandas as pd
    df = pd.read_csv(MODELSLOG)

    # To combine both L and R hippocampus predictions
    df["Subject"] = df["Subject"].str.replace('_mask_L.npy', '.npy')
    df["Subject"] = df["Subject"].str.replace('_mask_R.npy', '.npy')

    with open(FINALPRED, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Subject", "Diagnosis"])

        for name in df["Subject"].unique():
            g = df.groupby(['Subject'])
            print(g.get_group(name).Diagnosis.mode()[0])

            writer.writerow([name[:-4].split("/")[-1], g.get_group(name).Diagnosis.mode()[0]])

        f.close()

if __name__ == '__main__':
    start_time = time.time()

    removingAllExeptPredictingfiles(True)

    hippodeepToTwoHippos()
    for proj in PROJECTIONS:
        convertTo2D(proj)

        X, names = loadImgs()

        for cl in CLASSIFICATION:
            saveResults(predictMLPC(X, cl, proj), names, "MLPC", proj, cl)
            saveResults(predictLeNet(X, cl, proj), names, "LeNet", proj, cl)

    removingAllExeptPredictingfiles()
    generalise()

    print("--- %.2f seconds ---" % (time.time() - start_time))
