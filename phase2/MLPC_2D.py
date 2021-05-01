from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import statistics
from random import randrange
import numpy as np
from DataProcessing_2D import data_processing
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt

X, Y = data_processing(True, "Mean")


search_spaceMLPC = {
    "hidden_layer_sizes":Integer(100, 200, 10),
    "tol":Real(0.00001, 0.00005, 0.0001, 0.0005),
    "momentum":Real(0.95, 0.99, 0.01)
}
# clf = MLPClassifier(hidden_layer_sizes=100, max_iter=200, tol=0.0001, momentum=0.99).fit(X_train, y_train)
def doMLPClassifier(X, Y):
    print(X.shape, Y.shape)
    print(type(X), type(Y))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    def on_step(optim_result):
        score = bayesClf.best_score_
        print("Score: MLPC: ", score*100)
        if score == 1:
            print('Max Score Achieved')
            return True

    bayesClf = BayesSearchCV(MLPClassifier(max_iter=80, random_state=0), search_spaceMLPC,
                                n_iter=4, cv=5,
                                scoring="accuracy", return_train_score = False)
    bayesClf.fit(X_train, y_train, callback = on_step)
    y_pred = bayesClf.best_estimator_.predict(X_test)

    M = confusion_matrix(y_test, y_pred)
    print(M)

    mlp = MLPClassifier(**bayesClf.best_params_)
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 2
    N_BATCH = 32
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(mlp.score(X_test, y_test))

        epoch += 1

    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()

    logClassifier(SVC(), self.classes,
        metrics[0], metrics[1], metrics[2], metrics[3],
        metrics[4], metrics[5], metrics[6], bayesClf.best_params_)

# X, Y = shuffle(X, Y)
doMLPClassifier(X, Y)
