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
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit

def DT(X, y):

    HyperparametersSelection = False

    if HyperparametersSelection:
        from sklearn.model_selection import GridSearchCV
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer

        param_grid = {
        'n_estimators':[i for i in range(1, 100, 50)],
        'learning_rate':[i/10000 for i in range(1, 1000, 250)],
        'max_depth' : [1, 2],
        'max_features':[1, 2]
        }

        search_space = {
                "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
                "max_features": Categorical(['auto', 'sqrt','log2']),
                "min_samples_leaf": Integer(2, 10),
                "min_samples_split": Integer(2, 10),
                "n_estimators": Integer(100, 500),
                'learning_rate':Real(0.00001, 1)
            }

        clf = GradientBoostingClassifier(random_state=0)
        # clf1 = GridSearchCV(clf1, param_grid = param_grid, cv=4, return_train_score = False)
        clf1 = BayesSearchCV(clf, search_space, n_iter=15, cv=5,
                            scoring="accuracy",  return_train_score = False)
        # for param in clf.get_params().keys():
        #     print("nipples: ", param)

        clf1.fit(X, y)

        # print(clf.cv_results_)
        # df = pd.DataFrame(clf.cv_results_)
        # df.to_csv('DTHyperParameterTuning', index=False)
        print("score: ",clf1.best_score_)
        print("params: ",clf1.best_params_)



    else:
        # clf = GradientBoostingClassifier(n_estimators = 51, learning_rate = 0.0751, max_depth = 1, max_features = 1, random_state=0)
        clf = GradientBoostingClassifier(n_estimators = 226, max_depth = 6, learning_rate=0.7397015781008435, min_samples_split=10,
                                        min_samples_leaf = 6, max_features = 'auto', random_state=0)

        clf.fit(X, y)


        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print(matrix)

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
