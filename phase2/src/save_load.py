import csv
import os
import datetime
import matplotlib.pyplot as plt

from params import *

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def logClassifier(clf, iProj, accuracy, recall, precision, f1, AUC, report, matrix):

    utc_datetime = datetime.datetime.utcnow()
    time = utc_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # If log does not exist, create Column names
    if os.path.exists(modelsLog)!=True:
        with open(modelsLog, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Classifier", "Projection",
                            "Accuracy", "Recall", "Precision", "F-score",
                             "Report", "AUC", "Confusion Matrix", "Time"])
            f.close()

    # append existing 'modelsLog' file
    with open(modelsLog, mode='a+') as file_object:
        writer = csv.writer(file_object, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([clf, iProj, accuracy, recall, precision, f1,
                        report, AUC, matrix, time])

    file_object.close()

    return

def saveModel(model, name, iProj, sklrn):
    # sklearn model - MLPC
    if sklrn:
        from joblib import dump
        dump(model, uniquify(saveModelPath + name + iProj+'.joblib'))

    # Keras models
    else:
        model.save(uniquify(saveModelPath + name + iProj))

def plotter(history, clf=None, iProj=None):

    if type(history) is list:
        plt.plot(history[0], color='green', alpha=0.8, label='Train')
        plt.plot(history[1], color='magenta', alpha=0.8, label='Test')
        plt.title("Accuracy over epochs", fontsize=14)
        plt.xlabel('Epochs')
        plt.legend(loc='upper left')
        path = uniquify(pathForSavingPlots + clf + iProj + '_ACC.png')
        plt.savefig(path)
        # plt.show()

    else:
        #  "Accuracy"
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(clf + ' accuracy on ' + iProj)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        path = uniquify(pathForSavingPlots + clf + iProj + '_ACC.png')
        plt.savefig(path)
        # plt.show()
    plt.cla()   # Clear axis
