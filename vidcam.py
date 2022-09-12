import cv2
import os
import shutil

class VideoCapture:

    def record(name):
        capture = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        path = f'Interactive Test/video/{name.lower()}'
        try: 
            os.mkdir(path) 
        except OSError as error: 
            shutil.rmtree(path)
            os.mkdir(path) 
        
        path = f'Interactive Test/video/{name.lower()}/data.avi'
        videoWriter = cv2.VideoWriter(path, fourcc, 30.0, (640,480))

        while (True):
        
            ret, frame = capture.read()

            if ret:
                cv2.imshow('Need atleast 10s (Press ESC to finish)', frame)
                videoWriter.write(frame)
        
            if cv2.waitKey(1) == 27: # escape key to exit
                break
        
        capture.release()
        videoWriter.release()
        
        cv2.destroyAllWindows()

    def vid2frame(name):
        
        # Opens the Video file
        path_vid = f'Interactive Test/video/{name}/data.avi'
        path_frame = f'Interactive Test/frame_data/{name.lower()}/'
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
            cv2.imwrite(path_frame + f'{name.lower()}'+str(i)+'.jpg')
            #cv2.imwrite(path_frame + f'{name.lower()}'+str(i)+'.jpg', cv2.resize(frame, (224,224)))
            i+=1
        
        cap.release()
        
        return i
    
    def abandon(name):
        path = f'Interactive Test/vidcap/object_{name.lower()}'
        shutil.rmtree(path)