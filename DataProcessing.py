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
    df.dropna(inplace=True)

    # Remove rows with exclude =1
    df = df[df['Exclude'] != 1]
    return df

# def f_importances(coef, names)

def scaleData(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    newdf = scaler.transform(df)

    return newdf

# func should return labeled data(AD or HC marked as 1 - MCI removed)
def getXY():
    merged = pd.read_csv(PATH_Demo)
    # merged = pd.merge(merged, pd.read_csv(PATH_ASL), how='inner', on=['ID', 'ID'])
    # merged = pd.merge(merged, pd.read_csv(PATH_MadeUp), how='inner', on=['ID', 'ID'])
    merged = pd.merge(merged, pd.read_csv(PATH_Neuro), how='inner', on=['ID', 'ID'])
    merged = pd.merge(merged, pd.read_csv(PATH_sMRI), how='inner', on=['ID', 'ID'])

    clear_data(merged)


    AD = merged[['AD']].values
    MCI =  merged[['MCI']].values
    print((AD), len(MCI))
    merged = merged.drop(['ID','AD', 'MCI', 'Exclude'], axis=1)

    # print(merged.columns.tolist()) #all elements included in classification

    X = merged.values
    Y = np.asarray([int(not AD[i] and not MCI[i]) for i in range(len(AD))])


    X=scaleData(X)

    return X, Y
