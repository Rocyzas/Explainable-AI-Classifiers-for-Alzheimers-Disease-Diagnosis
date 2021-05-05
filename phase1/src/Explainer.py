import lime
from lime.lime_tabular import LimeTabularExplainer
from IPython.display import HTML
from sklearn.utils import shuffle
import math

from save_load import loadModel, uniquify, save_predictions
from params import *

def explainLIME(rows, name, classes, XY):

    print("Starting to Explain Model ", name, " on ", classes)

    try:
        clf = loadModel(name+"-"+classes)
    except FileNotFoundError:
        print(name ," Model for " ,classes ," not found")
        exit()

    X, y = shuffle(XY[0], XY[1])
    columns = XY[2]

    # already scaled using getXY (explainer.py)
    rows = XY[3]

    yn = clf.predict(rows)
    for i in range(len(yn)):
        if classes=="MCI_AD":
            if yn[i]==0:
                prediction = "Mild Cognitive Impairment"
            else:
                prediction = "Alzheimer's Disease"
        elif classes=="HC_AD":
            if yn[i]==0:
                prediction = "Healthy Case"
            else:
                prediction = "Alzheimer's Disease"
        elif classes=="HC_MCI":
            if yn[i]==0:
                prediction = "Healthy Case"
            else:
                prediction = "Mild Cognitive Impairment"
        elif classes=="MULTI":
            if yn[i]==0:
                prediction = "Healthy Case"
            elif yn[i]==1:
                prediction = "Mild Cognitive Impairment"
            else:
                prediction = "Alzheimer's Disease"

        print("PREDICTION ", i, "-", prediction)
        save_predictions(i, prediction, classes, clf)

    if classes!="MULTI":
        class1 = str(classes).partition('_')[0]
        class2 = str(classes).partition('_')[2]
        class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]
    else:
        class_names = ["HC", "MCI", "AD"]

    explainer = LimeTabularExplainer(X, mode = 'classification',
                                        training_labels=y,
                                        feature_selection= 'auto',
                                        class_names=class_names,
                                        feature_names = columns,
                                        discretize_continuous=True)

    ExplainPath_Specific = ExplainPath + name + "_" + classes

    try:
        os.mkdir(ExplainPath_Specific)
    except OSError:
        print ("Creation of the directory %s failed or already exist" % ExplainPath_Specific)

    if len(class_names)==2:
        toplabs=1
    else:
        toplabs=1

    for i in range(len(rows)):
        explanation = explainer.explain_instance(rows[i],
                                    clf.predict_proba,
                                    top_labels = toplabs,
                                    num_features=5*math.floor(math.log(len(columns), 2)))

        html_data = explanation.as_html()
        HTML(data=html_data)

        explanation.save_to_file(uniquify(ExplainPath_Specific  + "/" + str(i) + "_classif_explanation.html"))

# make thjis one called fro LIME
def explainELI5(rows, name, classes, XY):
    import eli5
    import eli5.sklearn
    from eli5 import show_weights
    from eli5.sklearn import PermutationImportance
    from eli5.sklearn import explain_prediction_linear_classifier
    from eli5 import show_prediction

    cols = XY[2]

    clf = loadModel(name+"-"+classes)

    ExplainPath_Specific = ExplainPath + name + "_" + classes

    weights = eli5.explain_weights(clf, feature_names=cols, top=(len(cols)+1))
    predictionA = eli5.explain_prediction(clf, XY[0][0], feature_names=cols, top=(len(cols)+1))
    predictionB = eli5.explain_prediction(clf, XY[0][1], feature_names=cols, top=(len(cols)+1))

    html = eli5.format_as_html(weights)
    with open(uniquify(ExplainPath_Specific  + "/" + classes +"_ELI5_WEIGHTS.html"), 'w') as f:
        f.write(html)
        f.close()

    html = eli5.format_as_html(predictionA)
    with open(uniquify(ExplainPath_Specific  + "/" + classes +"_ELI5_PREDICTION_A.html"), 'w') as f:
        f.write(html)
        f.close()
    html = eli5.format_as_html(predictionB)
    with open(uniquify(ExplainPath_Specific  + "/" + classes +"_ELI5_PREDICTION_B.html"), 'w') as f:
        f.write(html)
        f.close()
