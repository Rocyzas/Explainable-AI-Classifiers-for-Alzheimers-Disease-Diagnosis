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

X, Y = data_processing(False, "Mean")
def scaleData(X, Y):

    Y = np.expand_dims((Y), axis=-1)
    # X = np.expand_dims((X), axis=-1)

    # print(type(X), type(Y))
    max_val = X.max()
    # print(X.max())
    # exit()
    # print(X.shape[1])
    # exit()

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

X, Y = scaleData(X, Y)


def createModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1],X.shape[2], 1)))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))


    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def createModelSiameseNetworks():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 2), padding="same", activation="relu", input_shape=(X.shape[1],X.shape[2],1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (2, 2), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# model = createModel()
def createModelLeNet():
    model = models.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(2,2), activation='relu', input_shape=(X.shape[1],X.shape[2],1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters=16, kernel_size=(2,2), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=3, activation = 'softmax'))

    # model.add(layers.Conv2D(filters=6, kernel_size=(2,2), activation='relu', input_shape=(60,60,1)))
    # model.add(layers.AveragePooling2D())
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Conv2D(filters=16, kernel_size=(2,2), activation='relu'))
    # model.add(layers.AveragePooling2D())
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=64, activation='relu'))
    # model.add(layers.Dense(units=84, activation='relu'))
    # model.add(layers.Dense(units=3, activation = 'softmax'))

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def createModelAlexNet():

    model = models.Sequential([
        layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(X.shape[1],X.shape[2], 1)),#277 277 3
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # exit()
    return model

model = createModelAlexNet()

# import random
# random.shuffle(Y)
X_train_P, X_test_P, Y_train_P, Y_test_P = train_test_split(X, Y, test_size=0.09, random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(X_train_P, Y_train_P, test_size=0.3, random_state=123)

history = model.fit(X_train, Y_train,
                    batch_size=32,
                    epochs=2,
                    validation_data=(X_test, Y_test))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ynew = model.predict_classes(X_test_P)
print(confusion_matrix(Y_test_P, ynew))
print(accuracy_score(Y_test_P, ynew))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/media/rokas/HDD/Phase2/Metrics/CNN_LeNetACCmn.png')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/media/rokas/HDD/Phase2/Metrics/CNN_LeNetLOSSmn.png')
plt.show()

# print("IMPORTED")
# pathL = '/media/rokas/HDD/ADNI/Processed2DHippocampalLeftSorted'
# pathR = '/media/rokas/HDD/ADNI/Processed2DHippocampalRightSorted'
#
# pathLabels = "/media/rokas/HDD/ADNI/ADNI1_Screening_1.5T_3_21_2021.csv"
#
# arrayOfNPY_Left = []
# arraySubjectLeft = [] #Names of subject
#
# arrayOfNPY_Right = []
# arraySubjectRight = [] #Names of subject
#
# arrayLabels = []
# arraySubjects = []
#
# with open(pathLabels) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     next(csv_reader) #skipping first line
#     seen = []
#     for row in csv_reader:
#         # not dict, because multiple keys appears
#
#         if row[1] in seen: continue # skip duplicate
#
#         seen.append(row[1])
#         arraySubjects.append(row[1])
#         arrayLabels.append(row[2])
#
# # Getting data from Left hippocampal area
# for filePath in sorted(glob.glob(pathL + "/*")):
#     splt_fileName = '_'.join(((filePath.rsplit('/', 1)[1]).split('.')[0]).split('_')[1:4])
#     arrayOfNPY_Left.append(np.load(filePath))
#     arraySubjectLeft.append(splt_fileName)
#
# # Getting data from Right hippocampal area
# for filePath in sorted(glob.glob(pathR + "/*")):
#     splt_fileName = '_'.join(((filePath.rsplit('/', 1)[1]).split('.')[0]).split('_')[1:4])
#     arrayOfNPY_Right.append(np.load(filePath))
#     arraySubjectRight.append(splt_fileName)
#
# # print(len(arrayOfNPY_Left), len(arrayOfNPY_Right), len(arraySubjectLeft), len(arraySubjectRight))
#
# # Mapping Labels and Data together (indirectly, since key values are repetative, but too importand to drop out)
# X=[] #data
# Y=[] #labels
#
# for Data in range(len(arraySubjectLeft)):
#     # Getting data all together (left and right hippos)
#     '''LEFT'''
#     for Label in range(len(arraySubjects)):
#         if arraySubjectLeft[Data] == arraySubjects[Label]:
#             X.append(arrayOfNPY_Left[Data])
#             Y.append(arrayLabels[Label])
#             # print(m, " - ", X[Data].shape, "   =", X[Data].mean())
#
#     '''RIGHT'''
# for Data in range(len(arraySubjectRight)):
#     for Label in range(len(arraySubjects)):
#         if arraySubjectRight[Data] == arraySubjects[Label]:
#             X.append(arrayOfNPY_Right[Data])
#             Y.append(arrayLabels[Label])
#
# # one hot encode
# ''' CIA GALI EILISKUMAS SUSIPIST ATSARGIAI'''
# encoder = LabelEncoder()
# encoder.fit(Y)
# Y = encoder.transform(Y)
#
#
# ''' ============= CNN MODEL ============= '''
#
# x_max = max(X[i].shape[0] for i in range(len(X)))
# y_max = max(X[i].shape[1] for i in range(len(X)))
# print(x_max, y_max)
#
# Y1 = []
# for y in Y:
#     Y1.append([y])
# Y=np.array(Y1)
#
# X = np.array(X)
# X = np.expand_dims(X, axis=-1)
# # X = np.expand_dims(X, -1)
#
# # def create_model_CNN():
# #     mar = (x_max, y_max, 1)
# #     print(type(mar))
# #     print(type(np.array(X).shape[1:]))
# #     # print()
# #     model = models.Sequential()
# #     model.add(layers.Conv2D(32, (2, 2), activation='softmax', input_shape=mar)  )
# #     model.add(layers.MaxPooling2D((2, 2)))
# #     model.add(layers.Conv2D(16, (3, 3), activation='softmax'))
# #     model.add(layers.MaxPooling2D((2, 2)))
# #     model.add(layers.Conv2D(8, (3, 3), activation='softmax'))
# #     model.add(layers.MaxPooling2D((2, 2)))
# #     model.add(layers.Conv2D(16, (3, 3), activation='softmax'))
# #
# #     model.add(layers.Flatten())
# #     model.add(layers.Dense(8, activation='softmax'))
# #     model.add(layers.Dense(3))
# #
# #     model.summary()
# #
# #     model.compile(optimizer='adam',
# #                   loss='categorical_crossentropy',
# #                   metrics=['accuracy'])
# #
# #     return model




#
# def create_model_ANN():
#     model = models.Sequential()
#     model.add(layers.Dense(64, input_dim=(3600), activation='relu'))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(3, activation='sigmoid'))
#
#     model.summary()
#
#     model.compile(loss='categorical_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])
#     return model
#
#
# Y = np_utils.to_categorical(Y, 3)
#
# def create_model_ANN_example():
#
#     from keras.layers import Activation, Dense, Dropout
#     from keras.models import Sequential
# # building a linear stack of layers with the sequential model
#     model = Sequential()
#     model.add(Dense(512, input_shape=(3600,)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
#
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
#     return model
# # model = create_model_CNN()
# model = create_model_ANN_example()
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
#
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
#
# print(type(X_train))
# print(type(Y_train))
# print(type(X_test))
# print(type(Y_test))
# exit()
# history = model.fit(X_train, Y_train,
#           batch_size=128, epochs=20,
#           verbose=2,
#           validation_data=(X_test, Y_test))
# exit()
# exit()
# # print(type(X), type(Y))
# # print(X.shape, Y.shape)
# # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
# #
# # print(type(X_train), type(X_test), type(y_train), type(y_test))
# # print((X_train).shape, (X_test).shape, (y_train).shape, (y_test).shape)
#
# history = model.fit(X, Y, epochs=20, validation_split = 0.3)
#
# exit()
#
# y_pred = model.predict(X_test)
# # yp = [ max(y_pred[i]) for i in range(len(y_pred)) ]
#
# counter=0
# for e in range(len(y_test)):
#     print(y_test[e], y_pred[e])
#
# print(counter)
#
# # y_predicted = [1 if prediction >= 1 else 0 for prediction in y]
# # print(y)
# print(y_pred)
# print(test_y)
#
#
#
# print(X.shape)
# print(Y)
#
# print(model.predict(X))
# # for i in range(len(X)):
# #     score = model.evaluate(X[i], Y[i], batch_size=16)
# #     print(score[1], Y[i])
#
# # xt = np.asarray(X).astype('float32')
# # yt = np.asarray(Y).astype('float32')
# # history = model.fit(xt, yt, epochs=10, validation_data=0.2)
# # exit()
#
#
#
# '''
# tensorboard              1.13.1
# tensorboard-plugin-wit   1.7.0
# tensorflow               1.13.1
# tensorflow-estimator     1.13.0
# typing-extensions        3.7.4.2
# widgetsnbextension       3.5.1
#
# '''
#
