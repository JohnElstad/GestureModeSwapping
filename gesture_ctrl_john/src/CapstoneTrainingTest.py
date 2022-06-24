#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:38:28 2022

@author: John Elstad

This is the model used to train the data output by body_tracking_data_collection.py. To train different gestures,
change the ouptput shape of the model.
"""

import numpy as np
import csv
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

path = '/home/user/john_ws/src/gesture_ctrl_john/src/2DBodyTrainingDataV3.csv'
dataset = loadtxt(path, delimiter=',')
print(dataset[1])
print(np.shape(dataset[1]))
# split into input (X) and output (y) variables
X = dataset[:,0:36]
y = dataset[:,36]
print(np.shape(X))
print(np.shape(y))#Validation/training Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(X_train.shape)
print('X_test')
print(X_test)
print(y_train.shape)
print(y_test)
print(y_train)#One hot encode outputs
y_test = LabelBinarizer().fit_transform(y_test)
y_train = LabelBinarizer().fit_transform(y_train)
print(y_train.shape)
print(y_test.shape)



from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

def get_model():
    x = x_in = Input(shape=(36,), name="input")
    x = Dense(36, name="d1", activation="relu")(x)
    x = Dense(126, name="d2", activation="relu")(x)
    x = Dense(256, name="d3", activation="relu")(x)
    x = Dropout(0.2, name="drop2")(x)
    x = Dense(126, name="d4", activation="relu")(x)
    
    x = Dense(5, name="output")(x)
    x = Activation("softmax", name="s1")(x)

    m = Model(inputs=x_in, outputs=x)
    m.summary()
    
    return m

adam = Adam(lr=0.0001)

model = get_model()
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["acc"])

model.fit(X_train, y_train, epochs=300)
model.save('/home/user/john_ws/src/gesture_ctrl_john/src/BodyModelV4.h5')

model.load_weights('/home/user/john_ws/src/gesture_ctrl_john/src/BodyModelV4.h5')
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))