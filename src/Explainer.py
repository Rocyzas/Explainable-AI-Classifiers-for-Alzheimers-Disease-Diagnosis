from IPython.display import HTML

import lime
from lime.lime_tabular import LimeTabularExplainer

from DataProcessing import *
from Params import *
from LRclf import LR
from SVMclf import SVM
from DTclf import DT

def explain(rows, name, classes):

    print("Starting model ", name, " on ", classes)
    clf = loadModel(name+"-"+classes)

    # get dataframe of all the data for explainer
    df = getDf(False)

    XY = getXY(df, classes, rows)
    X, y = shuffleZipData(XY[0], XY[1])

    columns = list(rows.head(0))

    # rows = scaleData(XY[3])
    rows = XY[3]

    yn = clf.predict(rows)

    print("PREDICTION: ", yn)

    class1 = str(classes).partition('_')[0]
    class2 = str(classes).partition('_')[2]
    class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]

    explainer = LimeTabularExplainer(X, mode = 'classification', training_labels=y, feature_selection= 'auto',
                                               class_names=class_names, feature_names = columns,
                                                   discretize_continuous=False)

    print("STARTING EXPLAINATION")

    ExplainPath_Specific = ExplainPath + name + "_" + classes

    try:
        os.mkdir(ExplainPath_Specific)
    except OSError:
        print ("Creation of the directory %s failed" % ExplainPath_Specific)

    for i in range(len(rows)):
        explanation = explainer.explain_instance(rows[i], clf.predict_proba, top_labels=1, num_features=5)
        html_data = explanation.as_html()
        HTML(data=html_data)
        print(i, " is saved")
        print(ExplainPath_Specific)
        explanation.save_to_file(uniquify(ExplainPath_Specific  + "/" + str(i) + "_classif_explanation.html"))

def explainELI5(clf):
    import eli5
    return
    # from eli5.sklearn import explain_prediction
    # eli5.show_weights(clf.named_steps["model"])
