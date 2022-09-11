from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import tensorflow as tf
from pathlib import Path
import os

class object_preprocessing:
    def __init__(self, object_name):
        object_dir = Path("Interactive Test/vidcap/")
        self.object_list = []
        self.object_name = object_name
        for name in object_name:
            self.
            self.object_list.append(object_dir/f'{name}'/'frame')


    def load_data(self):
        