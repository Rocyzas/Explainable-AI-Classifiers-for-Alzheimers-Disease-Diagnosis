import time
import numpy as np
import statistics
import random
import sys
import os

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

N_ITER = 8
CV = 5

repetativeScore = 0

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def saveModel(model, modelName, classes):
    from joblib import dump
    # clf = load('Models/' + filename.joblib')
    name = modelName+"-"+classes
    dump(model, uniquify('Models/' + name +'.joblib'))

def DT(X, y, HyperparametersSelection):

    # TODO should use opt result as a treshold when to stop
    def on_step(optim_result):
        score = bayesClf.best_score_
        print("Score: DT: ", score*100)

    if HyperparametersSelection==1:

        search_space = {
                "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
                "max_features": Categorical(['auto', 'sqrt','log2']),
                "min_samples_leaf": Integer(1, 10),
                "min_samples_split": Real(0.001, 1),
                "n_estimators": Integer(100, 500),
                'learning_rate':Real(0.00001, 1)
            }

        bayesClf = BayesSearchCV(GradientBoostingClassifier(random_state=0), search_space,
                            n_iter=N_ITER, cv=CV,
                            scoring="accuracy",  return_train_score = False)

        bayesClf.fit(X, y, callback = on_step)

        # print("Best Score: ",clf1.best_score_*100 , "%")
        print("params: ",bayesClf.best_params_)

        # Giving tuned hyperparameters to a new model - gradient Boosting Decision Tree
        clf = GradientBoostingClassifier(**bayesClf.best_params_, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====DT=====\n Confusion Matrix")
        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))

        print(value, " ", bayesClf.best_score_*100)

        return clf


    else:

        # clf = GradientBoostingClassifier(n_estimators = 51, learning_rate = 0.0751, max_depth = 1, max_features = 1, random_state=0)
        clf = GradientBoostingClassifier(n_estimators = 316, max_depth = 18, learning_rate=0.2886211704574655, min_samples_split=0.8926590082327759,
                                        min_samples_leaf = 7, max_features = 'sqrt', random_state=0)

        clf.fit(X, y)


        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====DT=====\n Confusion Matrix")
        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value)

        return clf

def LR(X, y, HyperparametersSelection):

    def on_step(optim_result):
        score = bayesClf.best_score_
        print("Score: ", score*100)

    if HyperparametersSelection:
        gridSearch = {
            'C':[0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tol': [0.00001, 0.00005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            # 'tol': [i/1000 for i in range(1, 100)],
            'solver':['liblinear'],
            "penalty":['l1', 'l2']
        }
        search_space = {
                "penalty":Categorical(['l1', 'l2']),
                "solver":Categorical(['liblinear']),
                "tol":Real(1e-5, 1e-1),
                "C":Real(0.1, 10)
            }

        # gridClf = GridSearchCV(LogisticRegression(random_state=0), gridSearch, cv=CV,
        #                     scoring="balanced_accuracy",  return_train_score = False, verbose=1)
        bayesClf = BayesSearchCV(LogisticRegression(random_state=0), search_space, n_iter=N_ITER, cv=CV,
                            scoring="accuracy",  return_train_score = False)

        bayesClf.fit(X, y, callback=on_step)

        clf = LogisticRegression(**bayesClf.best_params_, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====LR=====\n Confusion Matrix")

        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", bayesClf.best_score_*100)

        return clf

    else:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=3,
                                    tol=0.05, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=5)
        matrix = confusion_matrix(y, y_pred)
        print("====LR=====\n Confusion Matrix")
        print(matrix)
        print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

        return clf

def SVM(X, y, HyperparametersSelection):

    def on_step(optim_result):
        score = bayesClf.best_score_
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

        bayesClf = BayesSearchCV(SVC(random_state=0), search_space, n_iter=N_ITER, cv=CV,
                            scoring="accuracy",  return_train_score = False)

        bayesClf.fit(X, y, callback=on_step)

        # print("Best Score: ",clf1.best_score_*100 , "%")
        # print("params: ",bayesClf.best_params_)
        clf = SVC(**bayesClf.best_params_, random_state=0, probability=True)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=4)
        matrix = confusion_matrix(y, y_pred)
        print("====SVM=====\n Confusion Matrix")

        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", bayesClf.best_score_*100)

        return clf
    else:
        clf = SVC(C=2.46, degree=2, gamma='scale', kernel='poly', tol=0.0248, probability=True)
        clf.fit(X, y)
        y_pred = cross_val_predict(clf, X, y, cv=5)
        matrix = confusion_matrix(y, y_pred)
        print("====SVM=====\n Confusion Matrix")
        print(matrix)
        print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

        return clf


# def selectBestClassifier():
#     from os import walk
#
#     my_files = []
#     for (dirpath, dirnames, filenames) in walk('Models/'):
#         my_files.extend(filenames)
#         break
#     print(my_files)
#
#     for name in my_files:
#         model = load(name)
#         model.get_
