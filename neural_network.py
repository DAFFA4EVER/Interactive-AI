import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.constraints import maxnorm
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import cv2

class neural_network_engine:
    def __init__(self, class_num, img_height, img_weight):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(img_height,img_weight,3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=class_num, activation='softmax'))
        self.model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x=x_train, y=y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    
    def model_summary(self):
        self.model.summary()
