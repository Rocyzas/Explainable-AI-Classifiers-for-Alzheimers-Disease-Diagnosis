import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

PATH_sMRI = os.getcwd() +  "/../VPHDARE_Alzheimers_Data/SheffieldProspective_sMRI.csv"
PATH_ASL = os.getcwd() +  "/../VPHDARE_Alzheimers_Data/SheffieldProspective_ASL.csv"
PATH_Demo = os.getcwd() +  "/../VPHDARE_Alzheimers_Data/SheffieldProspective_Demo.csv"
PATH_Neuro = os.getcwd() +  "/../VPHDARE_Alzheimers_Data/SheffieldProspective_Neuro.csv"
PATH_MadeUp = os.getcwd() +  "/../VPHDARE_Alzheimers_Data/MadeUp.csv"

def clear_data(df):
    # remove na rows

    df= df.replace(np.NaN, df.median())
    # df = df.fillna(method='ffill')
    # Remove rows with exclude =1
    # df.dropna(inplace=True)
    df = df[df['Exclude'] != 1]

    return df

# def f_importances(coef, names)

def scaleData(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    newdf = scaler.transform(df)

    return newdf

# func should return labeled data(AD or HC marked as 1 - MCI removed)
def getXY(scale, classes):
    merged = pd.read_csv(PATH_Demo)
    merged = pd.merge(merged, pd.read_csv(PATH_ASL), how='inner', on=['ID', 'ID'])
    merged = pd.merge(merged, pd.read_csv(PATH_Neuro), how='inner', on=['ID', 'ID'])
    merged = pd.merge(merged, pd.read_csv(PATH_sMRI), how='inner', on=['ID', 'ID'])

    merged = clear_data(merged)

    AD = merged[['AD']].values
    MCI =  merged[['MCI']].values


    if classes == 'HC_AD':
        merged = merged[merged['MCI'] != 1] # exclude MCI 1

    elif classes == 'MCI_AD':
        merged = merged[(merged['AD'] == 1) | (merged['MCI'] == 1)] # exclude HC


    merged = merged.drop(['ID','AD', 'MCI', 'Exclude'], axis=1)

    columns = merged.columns


    X = merged.values
    Y = np.asarray([int(not AD[i]) for i in range(len(merged))])

    if scale:
        X=scaleData(X)

    return X, Y, columns
