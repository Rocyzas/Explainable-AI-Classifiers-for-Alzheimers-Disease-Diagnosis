import time
import numpy as np
import statistics
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from matplotlib import pyplot as plt

from DataProcessing import *

import lime
from lime.lime_tabular import LimeTabularExplainer

def main():

    # Getting XY for training and testint
    XY = getXY(True)
    X = XY[0]
    y = XY[1]

    clf = LogisticRegression(solver='lbfgs', C=3, random_state=0)
    clf.fit(X, y)

    y_pred = cross_val_predict(clf, X, y, cv=5)
    matrix = confusion_matrix(y, y_pred)

    # predict_logreg = lambda x:clf.predict_proba(x).astype(float)
    feature_names = XY[2].values

    # print("START EXPLAINer")
    # explainer = LimeTabularExplainer(X, mode="regression",
    #                                             feature_names= feature_names)
    # idx = random.randint(1, len(XY[1]))
    # print("START EXPLAINation with idx: ", idx)
    # print(X[idx])
    #
    # explanation = explainer.explain_instance(X[idx], clf.predict, num_features=len(feature_names))
    # from IPython.display import HTML
    # html_data = explanation.as_html()
    # HTML(data=html_data)
    # print("no worries, this will be saved")
    # explanation.save_to_file("LOGREGclassif_explanation.html")

    print("===RESULT===")
    print(matrix)
    print(100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix)))

if __name__ == '__main__':
    start_time = time.time()

    main()

    print("--- %.2f seconds ---" % (time.time() - start_time))
