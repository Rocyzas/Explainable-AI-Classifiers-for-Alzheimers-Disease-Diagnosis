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


arr1 = []


def DT(X, y):
    '''
    cv5
    {'learning_rate': 0.001, 'max_depth': 1, 'max_features': 1, 'n_estimators': 1}	0.583333333333333	0.583333333333333	0.5	0.5	0.5	0.7
    {'learning_rate': 0.001, 'max_depth': 1, 'max_features': 1, 'n_estimators': 26}	0.583333333333333	0.583333333333333	0.5	0.5	0.5	0.7
    {'learning_rate': 0.001, 'max_depth': 1, 'max_features': 1, 'n_estimators': 51}	0.583333333333333	0.583333333333333	0.5	0.5	0.5	0.7
    {'learning_rate': 0.001, 'max_depth': 1, 'max_features': 1, 'n_estimators': 76}	0.583333333333333	0.583333333333333	0.5	0.5	0.5	0.7

    {'learning_rate': 0.0001, 'max_depth': 1, 'max_features': 1, 'n_estimators': 1}	0.583333333333333	0.583333333333333	0.5	0.5	0.5	0.65
    {'learning_rate': 0.0001, 'max_depth': 1, 'max_features': 1, 'n_estimators': 26}	0.583333333333333	0.583333333333333	0.5	0.5	0.5	0.65

    '''
    gridsearch = False

    if gridsearch:
        from sklearn.model_selection import GridSearchCV

        param_grid = {
        'n_estimators':[i for i in range(1, 100, 50)],
        'learning_rate':[i/10000 for i in range(1, 1000, 250)],
        'max_depth' : [1, 2],
        'max_features':[1, 2]
        }
        clf1 = GradientBoostingClassifier(random_state=0)
        clf = GridSearchCV(clf1, param_grid = param_grid, cv=4, return_train_score = False)
        clf.fit(X, y)
        # print(clf.cv_results_)
        df = pd.DataFrame(clf.cv_results_)
        df.to_csv('DTHyperParameterTuning', index=False)
        print("score: ",clf.best_score_)
        print("params: ",clf.best_params_)



    else:
        clf = GradientBoostingClassifier(n_estimators = 51, learning_rate = 0.0751, max_depth = 1, max_features = 2, random_state=0)

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

def main(argv):

    # Getting XY for training and testint !=DT since it does not require scaling
    XY = getXY(argv[0]!='DT', argv[1])

    if argv[0]=='DT':
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        model = DT(XY[0], XY[1])

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # nx = [item[0] for item in arr1]
        # nx1 = [item[1] for item in arr1]
        # nx2 = [item[2] for item in arr1]
        # print("nx:", len(nx), len(nx1), len(nx2))
        # plt.xlabel('estimators')
        # plt.ylabel('learning rate')
        #
        # ax.plot_trisurf(nx, nx1, nx2, linewidth=0, antialiased=True)
        # # ax.plot_surface(np.array(nx), np.array(nx1), np.array(nx2), color='b')
        # plt.show()

    elif argv[0]=='SVM':
        model = SVM(XY[0], XY[1])
    elif argv[0]=='LR':
        model = LR(XY[0], XY[1])
    else:
        exit(0)


    if argv[2]=='1':
        explain(XY, model)
        # print(str(model).partition('(')[0])


if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method' '0/1(explainability)'
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
