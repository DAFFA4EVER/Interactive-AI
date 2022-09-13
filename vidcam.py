from turtle import width
import cv2
import os
import shutil

class VideoCapture:

    def record(name, dir_path):
        capture = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        path = f'{dir_path}/video/{name.lower()}'
        try: 
            os.mkdir(path) 
        except OSError as error: 
            shutil.rmtree(path)
            os.mkdir(path) 
        
        frame_shape = [480, 640]

        width = frame_shape[1]
        height = frame_shape[0]

        color = (0,0,255)
        thickness = 5

        min_res = (min(frame_shape[:2]))
        max_res = (max(frame_shape[:2]))
        pad_res = (max_res-min_res)//2

        start_point = (pad_res,0)
        if(height > width):
            end_point = (min_res, width+pad_res)
        elif(height < width):
            end_point = (height+pad_res, min_res)

        path = f'{dir_path}/video/{name.lower()}/data.avi'
        videoWriter = cv2.VideoWriter(path, fourcc, 30.0, (640,480))

        while (True):
        
            ret, frame = capture.read()

            if ret:
                cv2.rectangle(frame, start_point, end_point, color, thickness)
                cv2.imshow('Need atleast 10s (Press ESC to finish)', frame)
                videoWriter.write(frame)
        
            if cv2.waitKey(1) == 27: # escape key to exit
                break
        
        capture.release()
        videoWriter.release()
        
        cv2.destroyAllWindows()

    def vid2frame(name, dir_path):
        # Opens the Video file
        path_vid = f'{dir_path}/video/{name}/data.avi'
        path_frame = f'{dir_path}/frame_data/{name.lower()}/'

        frame_shape = [480, 640]

        min_res = (min(frame_shape[:2]))
        max_res = (max(frame_shape[:2]))
        pad_res = (max_res-min_res)//2

        y=0
        x=pad_res
        h=min_res
        w=min_res

        try: 
            os.mkdir(path_frame) 
        except OSError as error: 
            shutil.rmtree(path_frame)
            os.mkdir(path_frame) 
        cap= cv2.VideoCapture(path_vid)
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(path_frame + f'{name.lower()}'+str(i)+'.jpg', frame[y:y+h, x:x+w])
            #cv2.imwrite(path_frame + f'{name.lower()}'+str(i)+'.jpg', cv2.resize(frame, (224,224)))
            i+=1
        
        cap.release()
            
        return i
    
    def abandon(name, dir_path):
        path = f'{dir_path}/video/object_{name.lower()}'
        shutil.rmtree(path)