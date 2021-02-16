import os
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from Params import *

def clear_data(df):
    # remove NaN rows and Exclude rows
    # df = df.replace(np.NaN, df.mean(numeric_only=True))
    df= df.replace(np.NaN, df.median())

    # Remove rows with exclude =1
    df = df[df['Exclude'] != 1]

    df = df.dropna(axis=1, how='all')


    return df

# def f_importances(coef, names)
def scaleData(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    newdf = scaler.transform(df)

    return newdf


def getXY(classes, fillData = pd.DataFrame()):
    merged = pd.read_csv(PATH_Demo)
    # merged = pd.read_csv(os.getcwd() +  "/../DataFiles/A.csv")
    merged = pd.merge(merged, pd.read_csv(PATH_ASL), how='inner', on=['ID', 'ID'])
    merged = pd.merge(merged, pd.read_csv(PATH_Neuro), how='inner', on=['ID', 'ID'])
    merged = pd.merge(merged, pd.read_csv(PATH_sMRI), how='inner', on=['ID', 'ID'])

    merged = clear_data(merged)

    if classes == 'HC_AD':
        merged = merged[merged['MCI'] != 1] # exclude MCI 1
        AD = merged[['AD']].values

    elif classes == 'MCI_AD':
        merged = merged[(merged['AD'] == 1) | merged['MCI'] == 1]
        MCI =  merged[['MCI']].values
        AD = merged[['AD']].values

    else:
        print("Required format; HC_AD or MCI_AD")
        exit()

    merged = merged.drop(listFeaturesRemove, axis=1)
    columns = pd.DataFrame(columns = list(merged.columns))

    if not fillData.empty:
        # Fill in empty spaces with mean values from FULL dataset
        fillData=fillData.fillna(merged.mean())
    else:
        # else, means no dataframe was given and hence a template has to be made
        columns.to_csv(saveColTemplate, sep=',', index=False)
        merged.to_csv(fulldata + classes + 'Data.csv', sep=',')

    X = merged.values
    Y = np.asarray([int(AD[i]) for i in range(len(merged))])

    # Concatinate
    X = np.append(X, fillData.to_numpy().reshape((fillData.shape[0], X.shape[1])), axis = 0)
    X=scaleData(X)
    # Split back / UNconcatinate
    fillData = X[len(X)-len(fillData):]
    X = np.delete(X, np.s_[len(X)-len(fillData):len(X)], axis = 0)


    return X, Y, columns, fillData
