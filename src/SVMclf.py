import numpy as np

from Params import *

from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix

from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from skopt.space import Real, Categorical, Integer


def SVM(X, y, HyperparametersSelection, classes):

    def on_step(optim_result):
        score = bayesClf.best_score_
        print("best score: ", score*100)
        if score==1:
            print("Max accuracy achieved")
            return True

    if HyperparametersSelection:
        search_space = {
            # "penalty":Categorical(['l1', 'l2', 'elasticnet', 'none']),
            # , 'poly', 'rbf', 'sigmoid']
            "kernel":Categorical(['poly', 'rbf', 'sigmoid', 'linear']),
            "degree":Integer(0, 10), #for poly only, ignored by other kernels
            "gamma":Real(1e-5, 1e-1),
            "coef0":Real(0, 1),
            "tol":Real(1e-5, 1e-1),
            "C":Real(1e-4, 1)
        }

        bayesClf = BayesSearchCV(SVC(random_state=0), search_space,
                                n_iter=N_ITER, cv=CV,
                                scoring="accuracy",  return_train_score = False)

        bayesClf.fit(X, y, callback=on_step)

        # print("Best Score: ",clf1.best_score_*100 , "%")
        print("params: ",bayesClf.best_params_)
        clf = SVC(**bayesClf.best_params_, random_state=0, probability=True)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=CV)
        matrix = confusion_matrix(y, y_pred)
        # 00-TN, 11 - TP, 10 -FN, 01 -FP
        print("====SVM=====\n Confusion Matrix")

        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", bayesClf.best_score_*100)

        logClassifier("SVM", classes, bayesClf.best_score_, matrix,  bayesClf.best_params_, y, y_pred)

        return clf

    else:
        # RBF fine
        # Poly fine
        # sigmoi fine
        clf = SVC(random_state=0, kernel='linear')
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=CV)
        matrix = confusion_matrix(y, y_pred)
        print("====SVM=====\n Confusion Matrix")
        print(matrix)
        print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

        return clf
