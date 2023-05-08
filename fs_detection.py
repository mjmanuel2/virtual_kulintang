import cv2
import numpy as np
import time
import sys
import faulthandler; faulthandler.enable()

"""
program: frame subtraction + basic color detection + drawing min. enclosing circle
- constraint in size of enclosing circle temporarily removed
"""

class sd_main(object):
    def __init__(self):
        '''sd_ver3.py'''
        self.frame_width = 640
        self.frame_height = 480
        self.center = (int(self.frame_width/2), int(self.frame_height/2))
        self.center_color = (255, 0, 0)
        self.bound_color = (0, 255, 0)
        self.radius = 59
        self.diameter = int(2*self.radius)
        self.patch_size = 30
        self.patch_half = int(self.patch_size/2)

        self.descriptor = []
        self.min_bgr = np.array([0,0,0])
        self.max_bgr = np.array([0,0,0])

        self.cam = None

        '''sd_ver4.py'''
        self.frame_count = 0
        self.frame_previous = np.array([0,0,0])
        self.frame_current = np.array([0,0,0])
        self.frame_rate = 0

        '''sd_ver5.py'''
        self.color_retrieved = False
    


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

        # Retrieve marker color
        while True:
            ret_val, frame = self.cam.read()
            mirrored = cv2.flip(frame, 1)
            image = cv2.circle(mirrored, self.center, self.radius + 1, self.center_color, 1)

            title = "Calibration"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self.frame_width, self.frame_height)
            cv2.setMouseCallback(title, self.get_color, mirrored)
            cv2.imshow(title, image)

            if self.color_retrieved == True:
                break
            if cv2.waitKey(1) == 27:
                break
        
        self.calibrated = True
        cv2.destroyAllWindows()



    def get_color(self, event, x, y, flags, image):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Crop image to N x N patch
            new_width = (int(self.center[0]-self.patch_half), int(self.center[0]+self.patch_half))
            new_height = (int(self.center[1]-self.patch_half), int(self.center[1]+self.patch_half))
            crop = image[new_height[0]:new_height[1], new_width[0]:new_width[1]]

            # Get descriptor top to bottom, left to right
            # --going through rows
            for y in range(crop.shape[0]):
                # --going through columns
                for x in range(crop.shape[0]):
                    if crop[y,x,1] != 0 and crop[y,x,2] != 0:
                        self.descriptor.append(crop[y,x])
            descriptor = np.array(self.descriptor)

            self.min_bgr[0] = np.min(descriptor[:,0])
            self.min_bgr[1] = np.min(descriptor[:,1])
            self.min_bgr[2] = np.min(descriptor[:,2])
            self.max_bgr[0] = np.max(descriptor[:,0])
            self.max_bgr[1] = np.max(descriptor[:,1])
            self.max_bgr[2] = np.max(descriptor[:,2])
                
            print("Minimum BGR: ", self.min_bgr)
            print("Maximum BGR: ", self.max_bgr)

            self.color_retrieved = True



    def blob_detection(self, frame, processed):
        # Color detection
        mask = cv2.inRange(processed, self.min_bgr, self.max_bgr)
        detected = cv2.bitwise_and(processed, processed, mask = mask)

        # Blob Detection
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations = 1)
        image, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Draw minimum enclosing circle
        # --approximate contours to polygons + get bounding circles
        contours_poly = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        # --draw polygonal contour + bounding circles
        for i in range(len(contours)):
            #cv2.drawContours(frame, contours_poly, i, self.bound_color)
            #if radius[i] >= self.radius and radius[i] < 200:
            cv2.circle(frame, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), self.bound_color, 1)
            cv2.circle(processed, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), self.bound_color, 1)



    def main_detection(self):
        ''' Main Code '''
        while True:
            self.frame_count += 1
            ret_val, frame = self.cam.read()
            frame = cv2.flip(frame, 1)

            if self.frame_count < 2:
                previous = frame
                current = frame
            
            elif self.frame_count >= 2:
                current = frame
            
            # Frame subtraction
            frame_processed = cv2.subtract(current, previous)

            # Blob detection
            #print(frame.shape, frame_processed.shape)
            self.blob_detection(frame, frame_processed)
            
            # Display processed
            display = np.concatenate((frame, frame_processed), axis=1)
            title = "Detection"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, int(self.frame_width*2), self.frame_height)
            cv2.imshow(title, display)

            if self.frame_count == 20:
                previous = current
                self.frame_count = 0
            if cv2.waitKey(1) == 27:
                break
        
        print("Quitting Detection...")
        self.cam.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    start = sd_main()
    start.calibration()
    start.main_detection()