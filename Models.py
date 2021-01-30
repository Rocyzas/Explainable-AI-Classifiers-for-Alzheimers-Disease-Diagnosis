import time
import numpy as np
import statistics
import random
import sys

# gradient boosting for regression in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load
# clf = load('filename.joblib')
# dump(clf, 'filename.joblib')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit

from matplotlib import pyplot as plt

from DataProcessing import *

def explain(XY, clf):
    from IPython.display import HTML
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    X = XY[0]
    y = XY[1]
    feature_names = XY[2].values

    print("START EXPLAINer")
    explainer = LimeTabularExplainer(X, mode="regression", feature_names= feature_names)
    idx = random.randint(1, len(y))
    print("START EXPLAINation with idx: ", idx)
    print(X[idx])

    explanation = explainer.explain_instance(X[idx], clf.predict, num_features=len(feature_names))

    html_data = explanation.as_html()
    HTML(data=html_data)
    print("no worries, this will be saved")
    explanation.save_to_file(str(clf).partition('(')[0] + "_classif_explanation.html")


def DT(X, y):
    clf = GradientBoostingClassifier(n_estimators=350, learning_rate=0.383, max_features=2, max_depth=2, random_state=0)
    clf.fit(X, y)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    matrix = confusion_matrix(y, y_pred)
    print(matrix)
    # arr1.append(i)
    value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
    print(value)

    return clf

def LR(X, y):
    clf = LogisticRegression(solver='lbfgs', C=3, random_state=0)
    clf.fit(X, y)

    y_pred = cross_val_predict(clf, X, y, cv=5)
    matrix = confusion_matrix(y, y_pred)
    print("===RESULT===")
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

    return clf

def SVM(X, y):
    clf = SVC(C=1.0, kernel='linear')
    clf.fit(X, y)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    matrix = confusion_matrix(y, y_pred)
    print("Linear")
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

    clf1 = SVC(C=1.0, kernel='poly')
    clf1.fit(X, y)
    y_pred = cross_val_predict(clf1, X, y, cv=5)
    matrix = confusion_matrix(y, y_pred)
    print("\n POLY:")
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

    return clf #olny the first one

def main(argv):

    # Getting XY for training and testint !=DT since it does not require scaling
    XY = getXY(argv[0]!='DT', 'MCI_AD')

    if argv[0]=='DT':
        model = DT(XY[0], XY[1])
    elif argv[0]=='SVM':
        model = SVM(XY[0], XY[1])
    elif argv[0]=='LR':
        model = LR(XY[0], XY[1])
    else:
        exit(0)


    if argv[1]=='1':
        explain(XY, model)
        # print(str(model).partition('(')[0])


if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 0/1(explainability)
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
