import time
import numpy as np
import statistics

# gradient boosting for regression in scikit-learn
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
# clf = load('filename.joblib')
# dump(clf, 'filename.joblib')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit

from matplotlib import pyplot as plt

from DataProcessing import *

def main():

    # Getting XY for training and testint
    XY = getXY(True)
    clf = LogisticRegression(solver='lbfgs', C=3, random_state=0).fit(XY[0], XY[1])

    y_pred = cross_val_predict(clf, XY[0], XY[1], cv=5)
    matrix = confusion_matrix(XY[1], y_pred)
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

if __name__ == '__main__':
    start_time = time.time()

    main()

    print("--- %.2f seconds ---" % (time.time() - start_time))
