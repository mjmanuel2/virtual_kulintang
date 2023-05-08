import cv2
import numpy as np
import time
import sys

"""
program: frame subtraction
- save two frames from camera feed
- implement subtraction
"""

class sd_main(object):
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.frame_count = 0
        self.frame_previous = np.array([0,0,0])
        self.frame_current = np.array([0,0,0])
        self.frame_processed = np.array([0,0,0])
        self.frame_rate = 0

    def calibration(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.frame_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.frame_height)

        # Calculate FPS
        num_frames = 30
        start = time.time()
        for i in range(num_frames):
            ret, frame = self.cam.read()
        end = time.time()
        self.frame_rate = int(num_frames/(end - start))
        print("Frame Rate: ", self.frame_rate)
    


    def frame_subtraction(self, previous, current):
        self.frame_processed = cv2.subtract(previous, current)

        title = "Processed"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.frame_width,self.frame_height)
        cv2.imshow(title, self.frame_processed)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()



    def blob_detection(self, image):
        pass



    def main_detection(self):
        while True:
        #while self.frame_count < 3:
            self.frame_count += 1
            ret_val, frame = self.cam.read()
            frame = cv2.flip(frame, 1)

            if self.frame_count < 2:
                previous = frame
                current = frame
            
            elif self.frame_count >= 2:
                current = frame
                #self.frame_subtraction(previous, current)
                #self.blob_detection(self.frame_processed)
            
            processed1 = cv2.subtract(previous, current)
            processed2 = cv2.subtract(current, previous)
            display = np.concatenate((frame, processed1, processed2), axis=1)

            title = "Detection"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, int(self.frame_width*2), self.frame_height)
            cv2.imshow(title, display)

            previous = current
            if cv2.waitKey(1) == 27:
                break
        
        print("Quitting Detection...")
        self.cam.release()
        cv2.destroyAllWindows()
            


    # ---------------------------------------------------------------
    # TEST FUNCTIONS
    def frame_values(self):
        while True:
            ret_val, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            title = "Detection"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self.frame_width, self.frame_height)
            cv2.imshow(title, frame)

            if cv2.waitKey(0) == 27:
                break

        print("Quiting...")
        print(type(frame))
        cv2.destroyAllWindows() 
    # ---------------------------------------------------------------



if __name__ == '__main__':
    start = sd_main()
    start.calibration()
    start.main_detection()

