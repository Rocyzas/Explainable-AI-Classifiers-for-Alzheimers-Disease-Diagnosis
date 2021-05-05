import numpy as np
from params import *
from save_load import logClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import cross_val_predict

# Hyperparameters tuning
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV

# Models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


class createModels(object):
    def __init__(self, Xtr, Xte, ytr, yte, classes=None, score = 0):
        # tr- training, te - testing data
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xte = Xte
        self.yte = yte
        self.classes = classes

        # from sklearn.metrics import SCORERS
        # print(sorted(SCORERS.keys()))

    # Get All metrics to log
    def calculateMetrics(self, y_pred):
        if self.classes!="MULTI":
            matrix = confusion_matrix(self.yte, y_pred)
            roc_auc = roc_auc_score(self.yte, y_pred)
            recall = recall_score(self.yte, y_pred)
            report = classification_report(self.yte, y_pred, labels=[0, 1])
            accuracy = accuracy_score(self.yte, y_pred)
            f1 = f1_score(self.yte, y_pred, average='binary')
            precision = precision_score(self.yte, y_pred, average='binary')

            return accuracy, recall, precision, f1, roc_auc, report, matrix

        else:
            matrix = confusion_matrix(self.yte, y_pred)
            F1Score=f1_score(self.yte, y_pred, average='macro')
            Precision=precision_score(self.yte, y_pred, average='macro')
            Recall=recall_score(self.yte, y_pred, average='macro')
            report = classification_report(self.yte, y_pred, labels=[0,1,2])
            accuracy = accuracy_score(self.yte, y_pred)

            return accuracy, Recall, Precision, F1Score, None, report, matrix

    # Gradient Boosted Decision Tree Classifier
    def DT(self):
        def on_step(optim_result):
            score = bayesClf.best_score_
            print("Score: DT: ", score*100)
            if score == 1:
                print('Max Score Achieved')
                return True

        bayesClf = BayesSearchCV(GradientBoostingClassifier(random_state=0), search_spaceDT,
                            n_iter=N_ITER, cv=CV,
                            scoring=scoringMetrics,  return_train_score = False)

        bayesClf.fit(self.Xtr, self.ytr, callback = on_step)

        y_pred = bayesClf.best_estimator_.predict(self.Xte)

        metrics = self.calculateMetrics(y_pred)
        logClassifier(GradientBoostingClassifier(), self.classes,
            metrics[0], metrics[1], metrics[2], metrics[3],
            metrics[4], metrics[5], metrics[6], bayesClf.best_params_, scoringMetrics)

        return GradientBoostingClassifier(**bayesClf.best_params_).fit(self.Xtr, self.ytr)

    # Support Vector Classifier
    def SVC(self):
        def on_step(optim_result):
            score = bayesClf.best_score_
            print("Score: SVC: ", score*100)
            if score == 1:
                print('Max Score Achieved')
                return True

        bayesClf = BayesSearchCV(SVC(random_state=0), search_spaceSVC,
                                n_iter=N_ITER, cv=CV,
                                scoring=scoringMetrics,
                                return_train_score = False)

        bayesClf.fit(self.Xtr, self.ytr, callback=on_step)

        y_pred = bayesClf.best_estimator_.predict(self.Xte)

        metrics = self.calculateMetrics(y_pred)
        logClassifier(SVC(), self.classes,
            metrics[0], metrics[1], metrics[2], metrics[3],
            metrics[4], metrics[5], metrics[6], bayesClf.best_params_, scoringMetrics)

        return SVC(**bayesClf.best_params_, probability=True).fit(self.Xtr, self.ytr)

    # Logistic Regression Classifier
    def LR(self):
        def on_step(optim_result):
            score = bayesClf.best_score_
            print("Score: LR: ", score*100)
            if score == 1:
                print('Max Score Achieved')
                return True

        bayesClf = BayesSearchCV(LogisticRegression(max_iter=100, random_state=0), search_spaceLR, cv=CV,
                            n_iter=N_ITER, scoring=scoringMetrics,  return_train_score = False)

        bayesClf.fit(self.Xtr, self.ytr, callback = on_step)
        y_pred = bayesClf.best_estimator_.predict(self.Xte)

        metrics = self.calculateMetrics(y_pred)
        logClassifier(LogisticRegression(), self.classes,
            metrics[0], metrics[1], metrics[2], metrics[3],
            metrics[4], metrics[5], metrics[6], bayesClf.best_params_, scoringMetrics)

        return LogisticRegression(**bayesClf.best_params_).fit(self.Xtr, self.ytr)
