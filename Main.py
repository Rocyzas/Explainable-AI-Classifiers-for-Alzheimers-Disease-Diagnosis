import time
import numpy as np
import statistics
import random
import sys
from math import sqrt

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DataProcessing import *
from Classifiers import *

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def explain(XY, clf, name, classes):
    from IPython.display import HTML
    import lime
    from lime.lime_tabular import LimeTabularExplainer

    X = XY[0]
    y = XY[1]
    y = y.flatten()

    feature_names = XY[2].values

    class1 = str(classes).partition('_')[0]
    class2 = str(classes).partition('_')[2]
    class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]

    explainer = LimeTabularExplainer(X, mode='regression',feature_selection= 'auto',
                                               class_names=class_names, feature_names = feature_names,
                                                   kernel_width=None,discretize_continuous=True)

    idx = random.randint(1, len(y))
    print("START EXPLAINation. idx: ", idx)
    print("Model: predicts: ", clf.predict(X[idx].reshape(1, -1)), " Actual: ", y[idx])

    explanation = explainer.explain_instance(X[idx], clf.predict, num_features=len(y), top_labels=5)

    # html_data = explanation.as_html()
    # HTML(data=html_data)
    # print("no worries, this will be saved")
    # # str(clf).partition('(')[0]
    # explanation.save_to_file(uniquify("ExplainHTML/"+ name  + "_" + classes + "_classif_explanation.html"))


def saveModel(model, name):
    from joblib import dump
    # clf = load('Models/' + filename.joblib')
    dump(model, uniquify('Models/' + name +'.joblib'))

def loadModel(name):
    from joblib import load
    model = load('Models/' + name + '.joblib')
    return model

def main(argv):

    # Getting XY for training and testint !=DT since it does not require scaling
    XY = getXY(argv[0]!='DT', argv[1])

    # newmod = loadModel(argv[0])
    # newmod.fit(XY[0], XY[1])
    #
    # y_pred = cross_val_predict(newmod, XY[0], XY[1], cv=4)
    # matrix = confusion_matrix(XY[1], y_pred)
    # print(matrix)
    #
    # value = 100*(matrix[0][0]+matrix[1][1])/(np.sum(matrix))
    # print(value)
    #
    # exit()
    if argv[0]=='DT':
        model = DT(XY[0], XY[1], int(argv[3]))

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
        model = SVM(XY[0], XY[1], int(argv[3]))
    elif argv[0]=='LR':
        model = LR(XY[0], XY[1], int(argv[3]))
    else:
        exit(0)

    # saveModel(model, argv[0])

    if argv[2]=='1':
        explain(XY, model, argv[0], argv[1])


if __name__ == '__main__':
    start_time = time.time()

    # python3 Model.py 'clasisfier' 'classification_method' '0/1(explainability)', True/False for selecting best hyper
    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
