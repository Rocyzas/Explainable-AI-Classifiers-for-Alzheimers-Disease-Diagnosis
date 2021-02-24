import numpy as np

from Params import *

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def DT(X, y, HyperparametersSelection, classes):

    # TODO should use opt result as a treshold when to stop
    def on_step(optim_result):
        score = bayesClf.best_score_
        print("Score: DT: ", score*100)
        if score == 1:
            print('Max Accuracy Achieved')
            return True

    if HyperparametersSelection==1:

        search_space = {
                "max_depth": Integer(5, 20), # values of max_depth are integers from 6 to 20
                "max_features": Categorical(['auto', 'sqrt','log2']),
                "min_samples_leaf": Integer(1, 10),
                "min_samples_split": Real(0.001, 1),
                "n_estimators": Integer(50, 500),
                'learning_rate':Real(0.000001, 1)
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

        y_pred = cross_val_predict(clf, X, y, cv=CV)
        matrix = confusion_matrix(y, y_pred)
        print("====DT=====\n Confusion Matrix")
        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))

        print(value, " ", bayesClf.best_score_*100)

        logClassifier("DT", classes, bayesClf.best_score_, matrix,  bayesClf.best_params_)

        return clf

    else:

        # clf = GradientBoostingClassifier(n_estimators = 51, learning_rate = 0.0751, max_depth = 1, max_features = 2, random_state=0)
        clf = GradientBoostingClassifier(n_estimators = 500, max_depth = 18, learning_rate=0.8210508648891495,
                                            min_samples_split=float(1), min_samples_leaf = 1,
                                                max_features = 'auto', random_state=0)

        clf.fit(X, y)


        y_pred = cross_val_predict(clf, X, y, cv=CV)
        # [print(y[i], y_pred[i]) for i in range(len(y))]
        matrix = confusion_matrix(y, y_pred)
        print("====DT=====\n Confusion Matrix")
        print(matrix)

        value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
        print(value)
        # clf = loadModel("DT-MCI_AD")
        # # clf.fit(X, y)
        # print(X)
        # yn = clf.predict(X)
        # # print(yn)
        # # y_pred = cross_val_predict(clf, X, y, cv=CV)
        # [print(int(y[i] == yn[i])) for i in range(len(y))]
        return clf
