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

#For Hyper parameters selecion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

N_ITER = 15
CV = 2

repetativeScore = 0

def DT(X, y, HyperparametersSelection):

    def on_step(optim_result):
        score = clf1.best_score_
        print("best score: ", score*100)

    if HyperparametersSelection==1:

        search_space = {
                "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
                "max_features": Categorical(['auto', 'sqrt','log2']),
                "min_samples_leaf": Integer(1, 10),
                "min_samples_split": Real(0.001, 1),
                "n_estimators": Integer(100, 500),
                'learning_rate':Real(0.00001, 1)
            }

        clf1 = BayesSearchCV(GradientBoostingClassifier(random_state=0), search_space,
                            n_iter=N_ITER, cv=CV,
                            scoring="accuracy",  return_train_score = False)

        clf1.fit(X, y, callback = on_step)

        # print("Best Score: ",clf1.best_score_*100 , "%")
        # print("params: ",clf1.best_params_)
        clf = GradientBoostingClassifier(**clf1.best_params_, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====DT=====")
        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", clf1.best_score_*100)

        return clf


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

def LR(X, y, HyperparametersSelection):

    def on_step(optim_result):
        score = clf1.best_score_
        print("best score: ", score*100)

    if HyperparametersSelection:
        search_space = {
                "penalty":Categorical(['l1', 'l2']),
                "solver":Categorical(['liblinear']),
                "tol":Real(1e-5, 1e-1),
                "C":Real(0.1, 10)
            }

        clf1 = BayesSearchCV(LogisticRegression(random_state=0), search_space, n_iter=N_ITER, cv=CV,
                            scoring="accuracy",  return_train_score = False)

        clf1.fit(X, y, callback=on_step)

        # print("Best Score: ",clf1.best_score_*100 , "%")
        # print("params: ",clf1.best_params_)
        clf = LogisticRegression(**clf1.best_params_, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====LR=====")

        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", clf1.best_score_*100)

        return clf

    else:
        clf = LogisticRegression(penalty='l2', solver='liblinear', C=7, tol=0.1, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=5)
        matrix = confusion_matrix(y, y_pred)
        print("===RESULT===")
        print(matrix)
        print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

        return clf

def SVM(X, y, HyperparametersSelection):

    def on_step(optim_result):
        score = clf1.best_score_
        print("best score: ", score*100)

    if HyperparametersSelection:
        search_space = {
                # "penalty":Categorical(['l1', 'l2', 'elasticnet', 'none']),
                "kernel":Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                "degree":Integer(0, 10), #for poly only, ignored by other kernels
                "gamma":Real(1e-5, 1e-1),
                "coef0":Real(0, 1),
                "tol":Real(1e-5, 1e-1),
                "C":Real(1e-4, 1e+4)
            }

        clf1 = BayesSearchCV(SVC(random_state=0), search_space, n_iter=N_ITER, cv=CV,
                            scoring="accuracy",  return_train_score = False)

        clf1.fit(X, y, callback=on_step)

        # print("Best Score: ",clf1.best_score_*100 , "%")
        # print("params: ",clf1.best_params_)
        clf = SVC(**clf1.best_params_, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====SVM=====")

        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", clf1.best_score_*100)

        return clf
    else:
        clf = SVC(C=2.46, degree=2, gamma='scale', kernel='poly', tol=0.0248)
        clf.fit(X, y)
        y_pred = cross_val_predict(clf, X, y, cv=5)
        matrix = confusion_matrix(y, y_pred)
        print("Linear")
        print(matrix)
        print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

        return clf
