import lime
from lime.lime_tabular import LimeTabularExplainer
from IPython.display import HTML
import matplotlib.pyplot as plt

import math

from DataProcessing import *
from Params import *
from LRclf import LR
from SVMclf import SVM
from DTclf import DT

def explain(rows, name, classes):

    print("Starting to Explain Model ", name, " on ", classes)
    try:
        clf = loadModel(name+"-"+classes)
    except FileNotFoundError:
        print(name ," Model for " ,classes ," not found")
        exit()
    # get dataframe of all the data for explainer
    df = getDf(False)

    XY = getXY(df, classes, rows)
    X, y = shuffleZipData(XY[0], XY[1])

    columns = list(rows.head(0))

    # already scaled in getXY
    rows = XY[3]

    yn = clf.predict(rows)

    print("PREDICTION: ", yn)

    class1 = str(classes).partition('_')[0]
    class2 = str(classes).partition('_')[2]
    class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]

    explainer = LimeTabularExplainer(X, mode = 'classification',
                                        training_labels=y,
                                        feature_selection= 'auto',
                                        class_names=class_names,
                                        feature_names = columns,
                                        # categorical_features=[1],
                                        discretize_continuous=False)

    print("STARTING EXPLAINATION")

    ExplainPath_Specific = ExplainPath + name + "_" + classes

    try:
        os.mkdir(ExplainPath_Specific)
    except OSError:
        print ("Creation of the directory %s failed or already exist" % ExplainPath_Specific)

    for i in range(len(rows)):
        explanation = explainer.explain_instance(rows[i],
                                    clf.predict_proba,
                                    top_labels=1, # top_labels=1
                                    num_features=2*math.floor(math.log(len(columns), 2)))

        # explanationas = explainer.explain_instance(rows[2], clf.predict_proba,
        #                             top_labels = 1, num_features=len(columns)) # top_labels=1
        # For each feature f in a single i prediction
        # elist = explanation.as_list(label=yn[i])
        # for f in range(len(columns)):
        #     alist = list(elist[0])
        #     # alist[1]*=-1
        #     # print(type(list(explanation.as_list(label=yn[i])[0])[1]), list(explanation.as_list(label=yn[i])[0])[1])
        #     list(explanation.as_list(label=yn[i])[0])[1] = 5
        #     print(list(explanation.as_list(label=yn[i])[0])[1])
        #     exit()
        # exit()
        # explanation = explanationas.as_list()

        html_data = explanation.as_html()
        HTML(data=html_data)
        print(i, " is saved")
        explanation.save_to_file(uniquify(ExplainPath_Specific  + "/" + str(i) + "_classif_explanation.html"))
        # exit()
        # with plt.style.context("ggplot"):
        #     explanation.as_pyplot_figure()
        # print(explanation.as_list())
    # into one file but thats a pyplotfigure so i need html
    # sp_obj = submodular_pick.SubmodularPick(explainer, df_titanic[model.feature_name()].values, prob, num_features=5,num_exps_desired=10)
    # [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]

def feature_importance(name, classes):
        clf = loadModel(name+"-"+classes)

        numberOfImportatnFeatures_toShow = 10

        # I dont even need that, only lsit of COLUMNS is required
        df = getDf(False)
        XY = getXY(df, classes)
        X, y = shuffleZipData(XY[0], XY[1])

        if clf.__class__.__name__=="GradientBoostingClassifier":
            print("Feature importance on ", clf.__class__.__name__)
            # print("Feature importance works only with Tree Based classifiers,"\
            # " in this case with GradientBoostingClassifier")
            # exit()

            importances = clf.feature_importances_
            # print(importances)
            std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)
            # print(std)
            # exit()
            indices = np.argsort(importances)[::-1]

            cols = list(XY[2])
            fc=[]

            for i in indices:
                fc.append(cols[i])
            # print(X.shape, len(importances), len(indices))
            # exit()
        elif clf.__class__.__name__=="LogisticRegression":
            importances = clf.coef_[0]
            abs_weights = np.abs(importances)

            # indices = np.fliplr(np.argsort(abs_weights))
            indices = (-abs_weights).argsort()[:len(abs_weights)]

            cols = list(XY[2])
            fc=[]

            for i in indices:
                fc.append(cols[i])
            print(importances)
            print(indices)


            # not possible for non-linear kernel (might check why for the report)
        elif clf.__class__.__name__=="SVC" and clf.kernel=="linear":

            importances = clf.coef_[0]
            abs_weights = np.abs(importances)

            # indices = np.fliplr(np.argsort(abs_weights))
            indices = (-abs_weights).argsort()[:len(abs_weights)]

            cols = list(XY[2])
            fc=[]

            for i in indices:
                fc.append(cols[i])
            print(importances)
            print(indices)

        else:
            print("Classifier required: GradientBoostingClassifier or LogisticRegression or SVC(linear)")
            exit()

        plt.figure(figsize=(6,8))
        plt.title("Feature importances " + str(clf.__class__.__name__))
        # print(len(importances[0]), len(indices[0]))
        # exit()
        plt.bar(range(X.shape[1]), importances[indices], color="r",
            align="center")
        plt.xticks(range(X.shape[1]), fc, rotation='vertical')
        plt.subplots_adjust(bottom=0.25)
        plt.xlim([-0.5, numberOfImportatnFeatures_toShow])
        plt.show()


def explainELI5(rows, name, classes):
    import eli5
    print("IMPORT OF ELI5 SUCCESSFULL")
    # exit()
    clf = loadModel(name+"-"+classes)
    print(eli5.show_weights(clf,vec=1, top=20))
    '''
    html_data = explanation.as_html()
    HTML(data=html_data)
    print(i, " is saved")
    explanation.save_to_file(uniquify(ExplainPath_Specific  + "/" + str(i) + "_classif_explanation.html"))
'''
    exit()
