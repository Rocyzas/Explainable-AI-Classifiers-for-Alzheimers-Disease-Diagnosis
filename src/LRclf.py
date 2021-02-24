import numpy as np

from Params import *

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def LR(X, y, HyperparametersSelection, classes):

    def on_step(optim_result):
        score = bayesClf.best_score_
        print("Score: ", score*100)
        if score==1:
            print("Max accuracy achieved")
            return True

    if HyperparametersSelection:

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

        y_pred = cross_val_predict(clf, X, y, cv=CV)
        matrix = confusion_matrix(y, y_pred)
        print("====LR=====\n Confusion Matrix")

        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value, " ", bayesClf.best_score_*100)

        logClassifier("LR", classes, bayesClf.best_score_, matrix,  bayesClf.best_params_)
        return clf

    else:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=3,
                                    tol=0.05, random_state=0)
        clf.fit(X, y)

        y_pred = cross_val_predict(clf, X, y, cv=CV)
        matrix = confusion_matrix(y, y_pred)
        print("====LR=====\n Confusion Matrix")
        print(matrix)
        print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

        return clf
