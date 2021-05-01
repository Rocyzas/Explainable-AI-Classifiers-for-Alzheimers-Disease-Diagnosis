from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import statistics
from random import randrange
import numpy as np
from DataProcessing_3D import data_processing
from sklearn.utils import shuffle

X, Y = data_processing()

# mX = []
# for d in X:
#     mX.append(d.flatten())
# X = np.array(mX)
print(Y)
# Y=Y.ravel()


def doMLPClassifier(X, Y):
    from sklearn.neural_network import MLPClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf = MLPClassifier(hidden_layer_sizes=100, max_iter=200, tol=0.0001, momentum=0.99).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # counter=0
    # print(y_pred)
    # for inx in range(len(X_test)):
    #     print(y_test[inx], y_pred[inx])
    #     if y_test[inx] == y_pred[inx]:
    #         counter+=1

    # print(counter/len(y_pred))
    M = confusion_matrix(y_test, y_pred)
    print(clf.score(X_test, y_test))
    print(M)

X, Y = shuffle(X, Y)
doMLPClassifier(X, Y)
