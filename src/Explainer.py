from IPython.display import HTML

import lime
from lime.lime_tabular import LimeTabularExplainer

from DataProcessing import *
from Params import *
from LRclf import LR
from SVMclf import SVM
from DTclf import DT

def explain(rows, clf, name, classes):

    # for explainer
    XY = getXY(classes, rows)
    X = XY[0]
    y = XY[1]

    columns = list(rows.head(0))

    # rows = scaleData(XY[3])
    rows = XY[3]

    predict_logreg = lambda x: clf.predict_proba(x).astype(float)

    yn = clf.predict(rows)

    print("PREDICTION: ", yn)

    class1 = str(classes).partition('_')[0]
    class2 = str(classes).partition('_')[2]
    class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]

    explainer = LimeTabularExplainer(X, mode = 'classification', training_labels=y, feature_selection= 'auto',
                                               class_names=class_names, feature_names = columns,
                                                   kernel_width=None,discretize_continuous=True)

    print("STARTING EXPLAINATION")

    for i in range(len(rows)):
        explanation = explainer.explain_instance(rows[i], clf.predict_proba, top_labels=10)
        html_data = explanation.as_html()
        HTML(data=html_data)
        # print(i, "th is saved")
        explanation.save_to_file(uniquify(ExplainPath + name  + "_" + classes + "_classif_explanation.html"))

def explainELI5(clf):
    import eli5
    return
    # from eli5.sklearn import explain_prediction
    # eli5.show_weights(clf.named_steps["model"])
