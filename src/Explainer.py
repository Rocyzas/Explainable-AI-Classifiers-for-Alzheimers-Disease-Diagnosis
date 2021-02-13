from IPython.display import HTML

import lime
from lime.lime_tabular import LimeTabularExplainer

from DataProcessing import *
from Params import *
from LRclf import *
from SVMclf import *
from DTclf import *

ExplainPath = "/home/rokas/Documents/Leeds/3Year1Term_Works/IndividualProject/Source/ExplainHTML/"

def explain(rows, clf, name, classes):

    # for explainer
    XY = getXY(classes, rows)
    X = XY[0]
    y = XY[1]

    columns = list(rows.head(0))

    # dbr returning reiks xy[3] - filledrows rows
    # rows = rows.drop(rows.columns[[0]], axis=1)
    # print(type(rows))
    print(type(rows))
    # print("mano penis: ", type(X))
    rows = scaleData(XY[3])
    print(type(rows))
    yn = clf.predict(rows)
    # print(clf)


    print("PREDICTION: ", yn)

    class1 = str(classes).partition('_')[0]
    class2 = str(classes).partition('_')[2]
    class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]

    explainer = LimeTabularExplainer(X, mode = 'regression', training_labels=y, feature_selection= 'auto',
                                               class_names=class_names, feature_names = columns,
                                                   kernel_width=None,discretize_continuous=True)

    print("START EXPLAINATION")

    # print("nipas ",rows.iloc[0].to_numpy())
    explanation = explainer.explain_instance(rows[0], clf.predict, top_labels=10)
    html_data = explanation.as_html()
    HTML(data=html_data)
    print("th is saved")
    explanation.save_to_file(uniquify(ExplainPath + name  + "_" + classes + "_classif_explanationBYBYs.html"))


def explainELI5(clf):
    import eli5
    return
    # from eli5.sklearn import explain_prediction
    # eli5.show_weights(clf.named_steps["model"])
