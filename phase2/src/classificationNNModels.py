import numpy as np
import glob
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from DataProcessing_2D import data_processing
from skopt import BayesSearchCV

from sklearn.neural_network import MLPClassifier
# saving summary
from keras.utils import plot_model

from skopt.space import Real, Categorical, Integer

from params import *


class createNNModels(object):

    def __init__(self, X_train, X_test, y_train, y_test, intensityProjection=None):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.intensityProjection = intensityProjection

    def createModelLeNet(self, outputs=False):
        from tensorflow.keras.metrics import Accuracy
        model = models.Sequential()
        model.add(layers.Conv2D(filters=6, kernel_size=(2,2), activation='relu',
                    input_shape=(self.X_train.shape[1],self.X_train.shape[2],1)))
        model.add(layers.AveragePooling2D())
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(filters=16, kernel_size=(2,2), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        # model.add(layers.Dense(units=512, activation='relu'))# should not be too large 512is ok
        model.add(layers.Dense(units=512, activation='relu'))
        model.add(layers.Dense(units=112, activation='relu'))
        # model.add(layers.Dense(units=112, activation='relu')) # witg 256 is ok

        "OUTPUT UNITS (BINARY - 2, MULTI-CLASS - 3)"
        model.add(layers.Dense(units=int(outputs)+2, activation = 'softmax'))

        model.summary()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics="accuracy")

        plot_model(model, to_file=FILETOSAVESUMMARY_LENET, show_shapes=True,show_layer_names=True)
        # exit()
        history = model.fit(self.X_train, self.y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(self.X_test, self.y_test))


        return model, history



    def createModelMLPC(self):

        def on_step(optim_result):
            score = bayesClf.best_score_
            print("Score: MLPC: ", score*100)
            if score == 1:
                print('Max Score Achieved')
                return True

        search_spaceMLPC = {
        #     "hidden_layer_sizes":Integer(100, 200, 10),
            "tol":Real(0.00001, 0.0005),
            "momentum":Real(0.90, 0.99)
        }

        bayesClf = BayesSearchCV(MLPClassifier(max_iter=1000, random_state=0, hidden_layer_sizes=(128, 64)), search_spaceMLPC,
                                    n_iter=ITERS_MLPC, cv=5,
                                    scoring="accuracy", return_train_score = False)

        bayesClf.fit(self.X_train, self.y_train, callback = on_step)

        mlp = MLPClassifier(**bayesClf.best_params_)
        N_TRAIN_SAMPLES = self.X_train.shape[0]
        N_EPOCHS = EPOCHS
        N_BATCH = BATCH_SIZE
        N_CLASSES = np.unique(self.y_train)

        scores_train = []
        scores_test = []

        # EPOCH
        epoch = 0
        while epoch < N_EPOCHS_MLPC:
            print('epoch: ', epoch)
            # SHUFFLING
            random_perm = np.random.permutation(self.X_train.shape[0])
            mini_batch_index = 0
            while True:
                # MINI-BATCH
                indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
                mlp.partial_fit(self.X_train[indices], self.y_train[indices], classes=N_CLASSES)
                mini_batch_index += N_BATCH

                if mini_batch_index >= N_TRAIN_SAMPLES:
                    break

            # SCORE TRAIN
            scores_train.append(mlp.score(self.X_train, self.y_train))

            # SCORE TEST
            scores_test.append(mlp.score(self.X_test, self.y_test))

            epoch += 1

        return mlp, [scores_train, scores_test]
