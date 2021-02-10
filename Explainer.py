from IPython.display import HTML

import lime
from lime.lime_tabular import LimeTabularExplainer

from DataProcessing import *
from Classifiers import *


def explain(rows, columns, clf, name, classes):

    XY = getXY(True, "HC_AD")
    X = XY[0]
    y = XY[1]

    feature_names = columns
    # print("LEN:: ", len(feature_names), feature_names[0])

    # [print(x) for x in (feature_names)]
    # rows = rows.reshape(1,-1)
    print(type(rows), rows.shape)

    class1 = str(classes).partition('_')[0]
    class2 = str(classes).partition('_')[2]
    class_names = [str(classes).partition('_')[0], str(classes).partition('_')[2]]

    # explainer = LimeTabularExplainer(rows, mode='regression',feature_selection= 'auto',
    #                                            class_names=class_names, feature_names = feature_names,
    #                                                kernel_width=None,discretize_continuous=True)
    y=y.reshape(-1,1)

    explainer = LimeTabularExplainer(X, mode = 'regression', training_labels=y, feature_selection= 'auto',
                                               class_names=class_names, feature_names = feature_names,
                                                   kernel_width=None,discretize_continuous=True)
    # idx = random.randint(1, len(X))
    print("START EXPLAINATION")
    # print("Model: predicts: ", clf.predict(X[idx].reshape(1, -1)), " Actual: ", y[idx])

    # reshapedRow = rows[2].reshape(1,-1)
    # print((rows[2].shape))
    # def wrapped_fn(X):
    #     p = clf.predict(X).reshape(-1, 1)
    #     return np.hstack((1-p, p))

    for i in range(len(rows)):
        explanation = explainer.explain_instance(rows[i], clf.predict, top_labels=5)
        html_data = explanation.as_html()
        HTML(data=html_data)
        print(i, "th is saved")
        explanation.save_to_file(uniquify("ExplainHTML/"+ name  + "_" + classes + "_classif_explanation.html"))

def explainELI5(clf):
    import eli5
    return
    # from eli5.sklearn import explain_prediction
    # eli5.show_weights(clf.named_steps["model"])
