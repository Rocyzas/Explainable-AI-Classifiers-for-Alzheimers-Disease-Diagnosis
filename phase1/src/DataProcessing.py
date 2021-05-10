import os
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

from params import *

def clear_data(df):
    # remove NaN rows and Exclude rows
    df = df.replace(np.NaN, df.mean(numeric_only=True))
    # df= df.replace(np.NaN, df.median())

    # Remove rows with exclude =1
    if 'Exclude' in df:
        df = df[df['Exclude'] != 1]

    df = df.dropna(axis=1, how='all')

    # Exclude ID's that contains 'm', leave only baseline imgs(ADNI)
    df = df[~df.ID.str.contains('|'.join(['m']))]

    return df

def scaleData(df):
    scaler = StandardScaler()

    scaler.fit(df)

    # if not inverse:
    df = scaler.transform(df)

    return df

def normaliseShef(df):
    df = df.replace(np.NaN, df.mean(numeric_only=True))

    # scale every reagion except
    exceptList = ['ID', 'TotalICVolume']

    for column in df.drop(exceptList,axis=1):

        # normalise data to match ADNI dataset brain region values
        df[column] = (df[column]*valueOfBrainVolumeADNI)/df['TotalICVolume']

    # since we have already used it, we can equalise it to ADNI volume
    df['TotalICVolume'] = valueOfBrainVolumeADNI

    return df

def getDf():

    try:
        if whichDatasets=="ADNI" or whichDatasets=="BOTH":
            merged = pd.read_csv(PATH_Demo_ADNI)
            # merged = pd.merge(merged, pd.read_csv(PATH_Neuro_ADNI), how='inner', on=['ID', 'ID'])
            merged = pd.merge(merged, pd.read_csv(PATH_ASL_ADNI), how='inner', on=['ID', 'ID'])
            merged = pd.merge(merged, pd.read_csv(PATH_sMRI_ADNI), how='inner', on=['ID', 'ID'])
        if whichDatasets=="SHEFF" or whichDatasets=="BOTH":
            SheffsMRI = normaliseShef(pd.read_csv(PATH_sMRI_Sheffield))
            # Sheffield data
            mergedSheffield = pd.read_csv(PATH_Demo_Sheffield)
            # mergedSheffield = pd.merge(mergedSheffield, pd.read_csv(PATH_Neuro_Sheffield), how='inner', on=['ID', 'ID'])
            mergedSheffield = pd.merge(mergedSheffield, pd.read_csv(PATH_ASL_Sheffield), how='inner', on=['ID', 'ID'])
            mergedSheffield = pd.merge(mergedSheffield, SheffsMRI, how='inner', on=['ID', 'ID'])
        if whichDatasets=="BOTH":
            # merging both ADNI and Sheffield datasets
            merged = pd.concat([merged, mergedSheffield], ignore_index=True)
        elif whichDatasets=="SHEFF":
            merged = mergedSheffield
        elif whichDatasets!="ADNI":
            print("Error with data to be processed definition")
            exit()

        # merged = pd.read_csv(PATH_synthetic)
        # merged = pd.read_csv(PATH_F01)

    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Exception: Empty data or header is encountered")

    # merged = pd.read_csv(PATH_synthetic)
    merged = clear_data(merged)

    print("DATA MERGED")
    return merged


def getXY(merged, classes, fillData = pd.DataFrame()):

    if classes=='MULTI':
        mcilist=[]
        Y = []
        [mcilist.append((int(ad), int(mci))) for ad, mci in zip(merged[['AD']].values, merged[['MCI']].values)]

        for c in mcilist:
            if c[0]==0 and c[1]==0:
                Y.append(0)
            elif c[0]==0 and c[1]==1:
                Y.append(1)
            elif c[1]==0 and c[0]==1:
                Y.append(2)
            else:
                print("DATASET CORRUPTED. (cannot be both MCI and AD as 1)")
                exit()

    else:
        doesClassContainAD = True
        if classes == 'HC_AD':
            merged = merged[merged['MCI'] != 1] # exclude MCI 1
            AD = merged[['AD']].values

        elif classes == 'MCI_AD':
            merged = merged[(merged['AD'] == 1) | merged['MCI'] == 1]
            MCI =  merged[['MCI']].values
            AD = merged[['AD']].values

        elif classes == 'HC_MCI':
            merged = merged[merged['AD'] != 1] # exclude AD 1
            MCI =  merged[['MCI']].values
            doesClassContainAD = False

        if doesClassContainAD:

            Y = np.asarray([int(AD[i]) for i in range(len(merged))])

        elif doesClassContainAD == False:
            Y = np.asarray([int(MCI[i]) for i in range(len(merged))])


    merged = merged.drop(listFeaturesRemove, axis=1, errors='ignore')

    if filterSpecificFeatures!=[]:
        merged = merged.filter(filterSpecificFeatures)
    # print(merged)
    # exit()
    print(merged.shape)

    fillData=fillData.fillna(merged.mean(numeric_only=True))

    X = merged.values

    # Concatinate
    X = np.append(X, fillData.to_numpy().reshape((fillData.shape[0], X.shape[1])), axis = 0)

    '''SCALING'''
    X=scaleData(X)

    # Split back / UNconcatinate
    fillData = X[len(X)-len(fillData):]

    X = np.delete(X, np.s_[len(X)-len(fillData):len(X)], axis = 0)

    return X, Y, list(merged.columns), fillData, merged
