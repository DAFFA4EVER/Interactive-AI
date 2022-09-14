from pyexpat import model
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
    def build_layer(self, class_num, img_height, img_weight, path):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(img_height,img_weight,3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=class_num, activation='softmax'))
        self.model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model_path = f'{path}/saved_model'
        self.pred_path = f'{path}/predict_target'
        self.n_class = class_num

    def train(self, x_train, y_train, x_val, y_val, ephochs_total, batch_size):
        self.model.fit(x=x_train, y=y_train, epochs=ephochs_total, batch_size=batch_size, validation_data=(x_val, y_val))
        self.batch_size = batch_size
    
    def model_summary(self):
        print(self.model.summary())
    
    def model_evaluation(self, x_test, y_test):
        loss, acc = self.model.evaluate(x=x_test, y=y_test, batch_size=self.batch_size)
        print(f'Test Loss : {loss} | Test Accuracy : {acc*100}%')
    
    def prediction(self, pred_data, pred_filename, labels_name):
        prediction = self.model.predict(pred_data, batch_size=self.batch_size)
        prediction = np.argmax(prediction, axis=-1)
        print('[Prediction Result]')
        preview_path = self.pred_path
        for i in range(len(pred_filename)):
            img = cv2.imread(f"{preview_path}/{pred_filename[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_shape = img.shape[:2]
            if(frame_shape[0] != frame_shape[1]): # to make it square
                min_res = (min(frame_shape))
                max_res = (max(frame_shape))
                pad_res = (max_res-min_res)//2
                y=0
                x=pad_res
                h=min_res
                w=min_res
                if(frame_shape[0] > frame_shape[1]): # portrait
                    y=pad_res - pad_res//2 # portrait face usually top
                    x=0
                img = img[y:y+h, x:x+w]
            plt.imshow(img)
            plt.title(f"{labels_name[prediction[i]]}")
            plt.axis('off')
            plt.show()
        

    def confusion_matrix(self, x_data, y_data):
        # Get prediction
        pred = self.model.predict(x_data, batch_size=self.batch_size)
        pred = np.argmax(pred, axis=-1)

        # Original Labels
        labels = np.argmax(y_data, axis=-1)
        from sklearn.metrics import classification_report
        print(classification_report(labels, pred))
    
    def save_model(self):
        file_name = input("Saved the model as : ")
        self.model.save(filepath=f'{self.model_path}/{file_name}_{self.n_class}.h5')
        print(f"Model saved at : {self.model_path}/{file_name}_{self.n_class}.h5")
    
    def load_trained_model(self):
        model_list = os.listdir(self.model_path)
        status = 0
        self.n_class = 0
        if(len(model_list)!=0):
            print("\n[Available model]")
            for i in range(len(model_list)):
                print(f'{i} : {model_list[i]}')
            choose = int(input("Choose model : "))
            while((choose > len(model_list)) or (choose < 0)):
                choose = int(input("Choose model : "))
            
            self.model = load_model(filepath=f'{self.model_path}/{model_list[choose]}')
            status = 1
            print(f"\n{model_list[choose]} has been loaded")
        else:
            print("No available model")
        return status
