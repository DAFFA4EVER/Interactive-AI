from vidcam import VideoCapture

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