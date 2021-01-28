import time
import numpy as np
import statistics
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
    # X_train, X_test, y_train, y_test = train_test_split(XY[0], XY[1], test_size=0.25)

    clf = SVC(C=1.0, kernel='linear')
    y_pred = cross_val_predict(clf, XY[0], XY[1], cv=5)
    matrix = confusion_matrix(XY[1], y_pred)
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

    clf1 = SVC(C=1.0, kernel='poly')
    y_pred = cross_val_predict(clf1, XY[0], XY[1], cv=5)
    matrix = confusion_matrix(XY[1], y_pred)
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

    # clf2 = SVC(C=1.0, kernel='sigmoid')
    # clf2.fit(X_train, y_train.ravel())
    # scores = cross_val_score(clf2, X_train, y_train, cv=5)
    # arra2.append(100*statistics.mean(scores))
    #
    # clf3 = SVC(C=1.0, kernel='rbf')
    # clf3.fit(X_train, y_train.ravel())
    # scores = cross_val_score(clf3, X_train, y_train, cv=5)
    # arra3.append(100*statistics.mean(scores))

    # print("Lin Accuracy: ", 100*statistics.mean(scores))
    # f_importances(clf.coef_, ['Gender', 'Age'])


def mainlinreg():
    from sklearn.linear_model import LinearRegression
    XY = getXY(PATH_Demo)
    X_train = XY[0]
    X_test = XY[1]
    y_train = XY[2]
    y_test = XY[3]

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    plt.scatter(y_test, X_test,  color ='b')
    plt.plot( y_test, X_test,color ='k')

    plt.show()
    # scores = cross_val_score(reg, X_test, y_test, cv=10)
    # print("Lin Accuracy: ", 100*statistics.mean(scores))

if __name__ == '__main__':
    start_time = time.time()
    main()
    # mainlinreg();
    print("--- %.2f seconds ---" % (time.time() - start_time))
