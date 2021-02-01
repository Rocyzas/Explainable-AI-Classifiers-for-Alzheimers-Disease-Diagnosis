import time
import numpy as np
import statistics
import random
import sys

from matplotlib import pyplot as plt

from DataProcessing import *
from Classifiers import *

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

def saveModel():
    from joblib import dump, load
    # clf = load('filename.joblib')
    # dump(clf, 'filename.joblib')

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
