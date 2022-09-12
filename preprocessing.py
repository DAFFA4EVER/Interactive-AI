import pandas as pd
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class object_preprocessing:
    def __init__(self, available, frame_path, dir_path):
        self.object = available
        self.framePath = frame_path
        self.path = dir_path
        
    def split_data(self):
        print("Splitting the data")
        self.train_data = []
        self.train_label = []
        self.train_name = []
        self.test_data = []
        self.test_label = []
        self.test_name = []
        self.val_data = []
        self.val_label = []
        self.val_name = []
        for idx in range(len(self.object)):
            img_path = f'{self.framePath}/{self.object[idx]}'
            print(img_path)
            img_list = os.listdir(img_path)
            np.random.shuffle(img_list)
            train_list, val_list, test_list = np.split(np.array(img_list), [int(len(img_list)*0.7), int(len(img_list)*0.85)])
            train_list = [img_path+'/'+ data for data in train_list.tolist()]
            val_list = [img_path+'/'+ data for data in val_list.tolist()]
            test_list = [img_path+'/'+ data for data in test_list.tolist()]
            print("---------------------------------")
            print('Object Name : ', self.object[idx])
            print('Object Code : ', idx)
            print('Total Images: ', len(img_list))
            print('Training : ', len(train_list))
            print('Validation : ', len(val_list))
            print('Testing : ', len(test_list))
            for img in train_list:
                self.train_data.append(img)
                self.train_label.append(idx)
                self.train_name.append(self.object[idx])
            for img in test_list:
                self.test_data.append(img)
                self.test_label.append(idx)
                self.test_name.append(self.object[idx])
            for img in val_list:
                self.val_data.append(img)
                self.val_label.append(idx)
                self.val_name.append(self.object[idx])
        print("---------------------------------")
        flag = input("Save split result?(Yes/No) ")
        if(flag.lower() == "yes"):
            self.save_split_to_excel()
        else:
            print("The split result only for this run")

    def plot_preview(self):
        preview = self.ready_data
        plt.figure(figsize=(10,5))
        for j in range(len(preview)):
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                img = cv2.imread(f"{preview[j]['train'][i]}")
                img = cv2.resize(img, (224,224))
                plt.imshow(img)
                plt.title(f"{preview[j]['object']}")
                plt.axis('off')
            plt.show()
    
    def save_split_to_excel(self):
        print("Saving split data")
        try:
            train = pd.DataFrame(self.train_data)
            train.columns=['img']
            train['labels'] = self.train_label
            train['name'] = self.train_name
            train.to_excel(f'{self.path}/train_list.xlsx')

            test = pd.DataFrame(self.test_data)
            test.columns=['img']
            test['labels'] = self.test_label
            test['name'] = self.test_name
            test.to_excel(f'{self.path}/test_list.xlsx')

            val = pd.DataFrame(self.val_data)
            val.columns=['img']
            val['labels'] = self.val_label
            val['name'] = self.val_name
            val.to_excel(f'{self.path}/val_list.xlsx')
            print("Save Excel Success")
        except OSError as error: 
            print("Save Excel Failed")
    
    def load_excel_split(self):
        print("Trying to load split data")
        try:
            train_df = pd.DataFrame(pd.read_excel(f'{self.path}/train_list.xlsx'))
            test_df = pd.DataFrame(pd.read_excel(f'{self.path}/test_list.xlsx'))
            val_df = pd.DataFrame(pd.read_excel(f'{self.path}/val_list.xlsx'))

            self.train_data = train_df['img']
            self.train_label = train_df['labels'].to_numpy()
            
            self.test_data = test_df['img']
            self.test_label = test_df['labels'].to_numpy()

            self.val_data = val_df['img']
            self.val_label = val_df['labels'].to_numpy()

            print("Load Excel Success")
        except OSError as error: 
            print("Load Excel Failed")

    def prepare_data(self, img_height, img_weight):
        print("Preparing the data")
        self.y_train = to_categorical(self.train_label)
        self.y_test = to_categorical(self.test_label)
        self.y_val = to_categorical(self.val_label)
        data_list = [self.train_data, self.test_data, self.val_data]
        self.x_train = []
        self.x_test = []
        self.x_val = []
        n = 0
        for data in data_list:
            for i in range(len(data)):
                img = cv2.imread(data[i])
                img = cv2.resize(img, (img_height, img_weight))
                #data[i] = np.array(img.astype(np.float32)/255.)
                img = img.astype(np.float32)/255.
                if(n == 0):
                    self.x_train.append(img)
                elif(n == 1):
                    self.x_test.append(img)
                elif(n == 2):
                    self.x_val.append(img)
            n += 1

        self.x_train = np.array(self.x_train, dtype=np.float32)
        self.x_test = np.array(self.x_test, dtype=np.float32)
        self.x_val = np.array(self.x_val, dtype=np.float32)


    