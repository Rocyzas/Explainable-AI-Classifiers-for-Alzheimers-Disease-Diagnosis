from DataProcessing_2D import data_processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils


X, Y = data_processing(True, "Mean")
Y = np_utils.to_categorical(Y, 3)

print(X.shape, Y.shape)
print(type(X), type(Y))

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


    # X[inx] = X[inx]/max_val
X = scale(X, -1, 2)

# print(X)
# print(Y)

# import random
# random.shuffle(X)
# random.shuffle(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)


def createModel():
# building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(128, input_shape=(2184,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))


    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                # metrics=[tf.keras.metrics.Recall()],
                metrics = ['accuracy'],
                optimizer='sgd')
    model.summary()
    # exit()
    return model

model = createModel()

history = model.fit(X_train, Y_train,
          batch_size=32, epochs=1000,
          verbose=1,
          validation_data=(X_test, Y_test))

print(model.get_config())


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
