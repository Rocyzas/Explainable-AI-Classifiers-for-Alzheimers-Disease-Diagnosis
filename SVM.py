import time
import numpy as np
import statistics
from joblib import dump, load
import random
# clf = load('filename.joblib')
# dump(clf, 'filename.joblib')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit

from matplotlib import pyplot as plt

from DataProcessing import *
import lime
from lime.lime_tabular import LimeTabularExplainer

def main():

    # Getting XY for training and testint
    XY = getXY(True)
    # X_train, X_test, y_train, y_test = train_test_split(XY[0], XY[1], test_size=0.25)
    X = XY[0]
    y = XY[1]

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

    feature_names = XY[2].values

    # print("START EXPLAINer")
    # explainer = LimeTabularExplainer(X, mode="regression",
    #                                             feature_names= feature_names)
    # idx = random.randint(1, len(XY[1]))
    # print("START EXPLAINation with idx: ", idx)
    # print(X[idx])
    #
    # explanation = explainer.explain_instance(X[idx], clf1.predict, num_features=len(feature_names))
    # from IPython.display import HTML
    # html_data = explanation.as_html()
    # HTML(data=html_data)
    # print("no worries, this will be saved")
    # explanation.save_to_file("SVM1classif_explanation.html")

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %.2f seconds ---" % (time.time() - start_time))
