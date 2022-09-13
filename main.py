from vidcam import VideoCapture
import os
from preprocessing import object_preprocessing as op
from neural_network import neural_network_engine as nne
import pathlib

frame_data = f'{pathlib.Path().resolve()}/frame_data'
default_path = f'{pathlib.Path().resolve()}'

while(True):
    flag = input("Add new object?(Yes/No) ")
    if(flag.lower() == "no"):
        break
    while(flag.lower() != "yes"):
        flag = input("Add new object?(Yes/No) ")

    object_name = input("Input your object name : ")
    VideoCapture.record(object_name, default_path)
    count_frame = VideoCapture.vid2frame(object_name, default_path)

    if count_frame < 300:
        print(f"Not enough frame data to process!!! Frame Count : {count_frame} Needed : 300")
        VideoCapture.abandon(object_name, default_path)
    else:
        print(f"{object_name} has been created!!!")


available_object = os.listdir(frame_data)
print("\n[---Saved Object---]")
for i in range(len(available_object)):
    print(f"{i} : {available_object[i].capitalize()}")

class_len = len(available_object)

img_height = 128
img_weight = 128

pre_data = op(available_object, frame_data, default_path)
mode = int(input("\nWhich one do you prefer?\n1. Use new data\n2. Load existing data\nChoose : "))
if(mode==1):
    pre_data.split_data()
else:
    pre_data.load_excel_split()

pre_data.prepare_data(img_height, img_weight)

gas = nne(class_len, img_height, img_weight)
gas.train(pre_data.x_train, pre_data.y_train, pre_data.x_test, pre_data.y_test)