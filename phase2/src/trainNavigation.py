import numpy as np

from DataProcessing_2D import data_processing
from save_load import logClassifier, plotter, saveModel
from sklearn.model_selection import train_test_split
from classificationNNModels import createNNModels

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, confusion_matrix, roc_curve, auc


def performScaling(X, Y):
    Y = np.expand_dims((Y), axis=-1)

    print(X.shape, Y.shape)

    max_val = X.max()


    def scale(X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom

    for inx in range(len(X)):
        # X[inx] = X[inx]/max_val
        X[inx] = scale(X[inx], -1, 1)

    X = np.expand_dims(X, axis=-1)
    return X, Y


def evaluate(model, X, y, name, iProj, val=False):

    if val:
        ynew = model.predict(X)
    else:
        ynew = model.predict_classes(X)

    print("Confusion Matrix")
    matrix = confusion_matrix(y, ynew)
    print(matrix)

    print("Accuracy")
    acc = accuracy_score(y, ynew)
    print(acc)

    print("Recall")
    rec = recall_score(y, ynew, average='macro')
    print(rec)

    print("Precision")
    prec = precision_score(y, ynew, average='macro')
    print(prec)

    print("F-score")
    f1 = f1_score(y, ynew, average='macro')
    print(f1)

    print("AUC score")
    f1 = f1_score(y, ynew, average='macro')
    print(f1)

    print("AUC")
    fpr, tpr, thresholds = roc_curve(y, ynew, pos_label=2)
    AUC = auc(fpr, tpr)

    report = classification_report(y, ynew, labels=[0,1,2])

    logClassifier(name, iProj, acc, rec, prec, f1, AUC, report, matrix)
    return


def navigation(currentProj, typeExclude):
    # calling metjods fro class navigation
    # dealing with user input
    iProj=currentProj
    X, Y = data_processing(False, iProj, typeExclude)
    X, Y = performScaling(X, Y)

    # For final testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # For cross validation
    X_train_P, X_test_P, Y_train_P, Y_test_P = train_test_split(X_train, Y_train, test_size=0.25, random_state=2)


    ''' Now creating models '''
    modelCreatorObj = createNNModels(X_train_P, X_test_P, Y_train_P, Y_test_P)

    model, history = modelCreatorObj.createModelLeNet(typeExclude=="multi")
    evaluate(model, X_test, Y_test, "LeNet_"+typeExclude+"_", iProj)
    plotter(history, "LeNet_"+typeExclude+"_", iProj)
    saveModel(model, "LeNet_"+typeExclude+"_", iProj, False)


    X, Y = data_processing(True, iProj, typeExclude)
    # X, Y = performScaling(X, Y)
    # For final testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # For cross validation
    X_train_P, X_test_P, Y_train_P, Y_test_P = train_test_split(X_train, Y_train, test_size=0.25, random_state=2)

    modelCreatorObjforMLP = createNNModels(X_train_P, X_test_P, Y_train_P, Y_test_P)
    model, history = modelCreatorObjforMLP.createModelMLPC()
    evaluate(model, X_test, Y_test, "MLPC_"+typeExclude+"_", iProj, True)
    plotter(history, "MLPC_"+typeExclude+"_", iProj)
    saveModel(model, "MLPC_"+typeExclude+"_", iProj, True)
