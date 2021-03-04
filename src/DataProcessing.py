import os
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler


from Params import *

def shuffleZipData(X, y=[]):

    if y!=[]:
        combination = list(zip(X, y))
        random.shuffle(combination)
        X, y = zip(*combination)
    elif y==[]:
        combination = list(X)
        random.shuffle(combination)
        X = combination

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def clear_data(df):
    # remove NaN rows and Exclude rows
    df = df.replace(np.NaN, df.mean(numeric_only=True))
    # df= df.replace(np.NaN, df.median())

    # Remove rows with exclude =1
    if 'Exclude' in df:
        df = df[df['Exclude'] != 1]

    df = df.dropna(axis=1, how='all')

    # Exclude ID's that contains 'm'
    df = df[~df.ID.str.contains('|'.join(['m']))]

    return df

# def f_importances(coef, names)
def scaleData(df, inverse = False):
    # scaler = MinMaxScaler()
    # scaler = PowerTransformer()
    scaler = StandardScaler()

    scaler.fit(df)

    if not inverse:
        df = scaler.transform(df)
    elif inverse:
        df = scaler.inverse_transform(df)

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

# TODO: nereik nx tos value cia
def getDf(value = True):

    try:
        '''
        # ADNI data
        merged = pd.read_csv(PATH_Demo_ADNI)
        # merged = pd.merge(merged, pd.read_csv(PATH_Neuro_ADNI), how='inner', on=['ID', 'ID'])
        merged = pd.merge(merged, pd.read_csv(PATH_ASL_ADNI), how='inner', on=['ID', 'ID'])
        merged = pd.merge(merged, pd.read_csv(PATH_sMRI_ADNI), how='inner', on=['ID', 'ID'])

        # Normalising sMRI data values
        SheffsMRI = normaliseShef(pd.read_csv(PATH_sMRI_Sheffield))

        # Sheffield data
        mergedSheffield = pd.read_csv(PATH_Demo_Sheffield)
        # mergedSheffield = pd.merge(mergedSheffield, pd.read_csv(PATH_Neuro_Sheffield), how='inner', on=['ID', 'ID'])
        mergedSheffield = pd.merge(mergedSheffield, pd.read_csv(PATH_ASL_Sheffield), how='inner', on=['ID', 'ID'])
        mergedSheffield = pd.merge(mergedSheffield, SheffsMRI, how='inner', on=['ID', 'ID'])

        # merging both ADNI and Sheffield datasets
        merged = pd.concat([merged, mergedSheffield], ignore_index=True)
        '''
        merged = pd.read_csv(PATH_synthetic)

    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Exception: Empty data or header is encountered")
    # merged = pd.read_csv(PATH_synthetic)
    merged = clear_data(merged)

    print("DATA MERGED")
    return merged


def getXY(merged, classes, fillData = pd.DataFrame()):

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
        # AD = merged[['AD']].values #AD is 0 for healthy case
        doesClassContainAD = False


    # if True:
    merged = merged.drop(listFeaturesRemove, axis=1, errors='ignore')

    # merged = merged.filter(['RightParahippocampalGyrus', 'LeftParahippocampalGyrus',
    #                 ' Right Hippocampus', ' Left Hippocampus', ' Right PHG parahippocampal gyrus',
    #                 ' Left PHG parahippocampal gyrus', 'RightHippocampus', 'LeftHippocampus'])
    # merged = merged.filter([' Right PHG parahippocampal gyrus',
    #                 ' Left PHG parahippocampal gyrus', 'RightHippocampus', 'LeftHippocampus'])
    columns = pd.DataFrame(columns = list(merged.columns))
    # columns.to_csv(fulldata + "LAIKINAIVISICOLUMAI.csv", sep=',')

    # Do i need numeric_only=True??
    fillData=fillData.fillna(merged.mean(numeric_only=True))


    # Separate X(data) and Y(labels)
    X = merged.values
    if doesClassContainAD:
        Y = np.asarray([int(AD[i]) for i in range(len(merged))])

    elif doesClassContainAD == False:
        # else label y=1 when MCI=1 and 0 when its HC
        Y = np.asarray([int(MCI[i]) for i in range(len(merged))])

    # Concatinate
    X = np.append(X, fillData.to_numpy().reshape((fillData.shape[0], X.shape[1])), axis = 0)
    '''SCALLLERR'''
    X=scaleData(X)

    # Split back / UNconcatinate
    fillData = X[len(X)-len(fillData):]

    X = np.delete(X, np.s_[len(X)-len(fillData):len(X)], axis = 0)

    return X, Y, columns, fillData, merged
