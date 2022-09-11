from vidcam import VideoCapture
import os
from neural_network import object_preprocessing as op

while(True):
    flag = input("Add new object?(Yes/No) ")
    if(flag.lower() == "no"):
        break
    while(flag.lower() != "yes"):
        flag = input("Add new object?(Yes/No) ")

    object_name = input("Input your object name : ")
    VideoCapture.record(object_name)
    count_frame = VideoCapture.vid2frame(object_name)


    if count_frame < 300:
        print(f"Not enough frame data to process!!! Frame Count : {[count_frame]}")
        VideoCapture.abandon(object_name)
    else:
        print(f"Object {object_name} has been created!!!")

available_object = os.listdir(r'Interactive Test/vidcap')
print("Available Object :")
for i in range(len(available_object)):
    print(available_object[i][7:].capitalize())

op(available_object).load_data()