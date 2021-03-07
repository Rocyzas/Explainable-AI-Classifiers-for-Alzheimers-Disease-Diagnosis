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

def feature_importance(clf, name, classes, columns):

        numberOfImportatnFeatures_toShow = 10


        cols = list(columns)
        fc=[]
        # print("LENGTH OF THE WITHTS", len(clf.feature_importances_), clf.__class__.__name__)

        if clf.__class__.__name__=="GradientBoostingClassifier":
            print("Feature importance on ", clf.__class__.__name__)

            importances = clf.feature_importances_
            std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

            indices = np.argsort(importances)[::-1]

            for i in indices:
                fc.append(cols[i])

        elif clf.__class__.__name__=="LogisticRegression":
            importances = clf.coef_[0]
            abs_weights = np.abs(importances)

            # indices = np.fliplr(np.argsort(abs_weights))
            indices = (-abs_weights).argsort()[:len(abs_weights)]

            for i in indices:
                fc.append(cols[i])

            # not possible for non-linear kernel (might check why for the report)
        elif clf.__class__.__name__=="SVC" and clf.kernel=="linear":

            importances = clf.coef_[0]
            abs_weights = np.abs(importances)
            print("BSK PRIES:: ", len(importances), len(abs_weights))
            # print("BSK PRIES:: ", len(indices))
            # indices = np.fliplr(np.argsort(abs_weights))
            indices = (-abs_weights).argsort()[:len(abs_weights)]

            print(len(cols))
            print(len(indices))
            for i in indices:
                fc.append(cols[i])

        else:
            print("Classifier required: GradientBoostingClassifier or LogisticRegression or SVC(linear)")
            return

        plt.figure(figsize=(6,8))
        plt.title("Feature importances " + str(clf.__class__.__name__))
        # print(len(importances[0]), len(indices[0]))
        # exit()
        # print(X.shape[1], (columns), "sssss")
        print(type(columns), len(columns), type(importances[indices]), len(importances[indices]))
        plt.bar(range(len(columns)), importances[indices], color="r", align="center")
        plt.xticks(range(len(columns)), fc, rotation='vertical')
        plt.subplots_adjust(bottom=0.25)
        plt.xlim([-0.5, numberOfImportatnFeatures_toShow])


        try:
            path = modelsPath + "FeatureImportance"
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed or already exist" % path)

        path = path + "/" + classes
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed or already exist" % path)

        plt.savefig(uniquify(path  + "/" + name + "_Feature_Importance.png"))

        # plt.show()
        # plt.clf()
        # plt.close()
        plt.close('all')


def explainELI5(rows, name, classes):
    df = getDf(False)

    XY = getXY(df, classes, rows)
    # X, y = shuffleZipData(XY[0], XY[1])
    cols = XY[2]

    import eli5

    clf = loadModel(name+"-"+classes)

    ExplainPath_Specific = ExplainPath + name + "_" + classes

    weights = eli5.explain_weights(clf, feature_names=cols, top=len(cols))
    html = eli5.format_as_html(weights)

    # explanation.save_to_file()


    with open(uniquify(ExplainPath_Specific  + "/" + "_ELI5.html"), 'w') as f:
        f.write(html)
        f.close()
