import cv2
import numpy as np
import matplotlib as plt
import time

"""
program: image segmentation using rg chromaticity + blob detection
change log:
    04/12/23 @ blob detection: erosion instead of dilation (minimize white noise)
    04/13/23 @ blob detection: find largest circle + calculate bottom part
    04/28/23: non-parametric image segmentation
"""

class segmentation(object):
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.center = (int(self.frame_width/2), int(self.frame_height/2))
        self.radius = 59
        self.diameter = int(2*self.radius)

        self.center_color = (255, 0, 0)
        self.bound_color = (0, 255, 0)
        self.detection_point = np.array([0, 0])
        self.dp_update = False

        self.patch_size = 30
        self.patch_half = int(self.patch_size/2)
        self.patch_retrieved = False

        self.cam = None
        
        self.bins = 32
        self.matrix = np.zeros((32,32), dtype='int')
        self.patch = np.array([0,0,0])
        self.patch_r = np.array([0,0,0])
        self.patch_g = np.array([0,0,0])
        self.frame_r = np.array([0,0,0])
        self.frame_g = np.array([0,0,0])
        self.masked = np.array([0,0,0])



    def calibration(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,self.frame_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,self.frame_height)

        while True:
            ret_val, frame = self.cam.read()
            mirrored = cv2.flip(frame, 1)
            image = cv2.circle(mirrored, self.center, self.radius + 1, self.center_color, 1)

            title = "Calibration"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self.frame_width, self.frame_height)
            cv2.setMouseCallback(title, self.retrieve_patch, mirrored)
            cv2.imshow(title, image)

            if self.patch_retrieved == True:
                break
            if cv2.waitKey(1) == 27:
                break
        
        print("Quitting Calibration...")
        cv2.destroyAllWindows()
        self.calibrated = True



    def retrieve_patch(self, event, x, y, flags, image):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Crop image to N x N patch
            new_width = (int(self.center[0]-self.patch_half), int(self.center[0]+self.patch_half))
            new_height = (int(self.center[1]-self.patch_half), int(self.center[1]+self.patch_half))
            self.patch = image[new_height[0]:new_height[1], new_width[0]:new_width[1]]

            # RG chromaticity (normalized) of patch
            np.seterr(invalid='ignore')
            self.patch_r = self.patch[:,:,2] / self.patch.sum(axis=2)
            self.patch_g = self.patch[:,:,1] / self.patch.sum(axis=2)
            patch_r_int = (self.patch_r*(self.bins-1)).astype(int)
            patch_g_int = (self.patch_g*(self.bins-1)).astype(int)

            self.matrix, xbin, ybin = np.histogram2d(patch_g_int.flatten(), patch_r_int.flatten(), bins = self.bins, range = [[0,self.bins], [0, self.bins]])

            self.patch_retrieved = True
            print(self.patch.shape)



    def gaussian(self, channel, mean, std):
        '''
        probability equation
        '''
        np.seterr(invalid='ignore')
        return np.exp(-(channel-mean)**2/(2*std**2))*(1/(std*((2*np.pi)**0.5)))



    def image_segmentation(self, frame, mean = 1, std = 1):
        '''
        reference: https://towardsdatascience.com/image-processing-with-python-using-rg-chromaticity-c585e7905818
        '''

        # RG chromaticity of frame
        np.seterr(invalid='ignore')
        self.frame_r = frame[:,:,2] / frame.sum(axis=2)
        self.frame_g = frame[:,:,1] / frame.sum(axis=2)

        # Gaussian distribution
        std_patch_r = np.std(self.patch_r.flatten())
        mean_patch_r = np.mean(self.patch_r.flatten())

        std_patch_g = np.std(self.patch_g.flatten())
        mean_patch_g = np.mean(self.patch_g.flatten())

        masked_frame_r = self.gaussian(self.frame_r, mean_patch_r, std_patch_r)
        masked_frame_g = self.gaussian(self.frame_g, mean_patch_g, std_patch_g)
        final_mask = masked_frame_r * masked_frame_g

        final_mask = np.array(final_mask, dtype = "uint8")
        self.masked = cv2.bitwise_and(frame, frame, mask=final_mask)

        return final_mask
    


    def non_parametric_segmentation(self, frame):
        bins = 32

        # RG chromaticity of frame
        np.seterr(invalid='ignore')
        self.frame_r = frame[:,:,2] / frame.sum(axis=2)
        self.frame_g = frame[:,:,1] / frame.sum(axis=2)
        
        frame_r_int = (self.frame_r*(bins-1)).astype(int)
        frame_g_int = (self.frame_g*(bins-1)).astype(int)

        back_projection = np.zeros(self.frame_r.shape, dtype = 'uint8')
        for i in range(self.frame_r.shape[0]):
            for j in range(self.frame_r.shape[1]):
                back_projection[i,j] = self.matrix[frame_g_int[i,j], frame_r_int[i,j]]
        self.masked = cv2.bitwise_and(frame, frame, mask = back_projection)

        return back_projection



    def blob_detection(self, frame, masked, mask):
        self.dp_update = False

        # Blob Detection
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.erode(mask, kernel, iterations = 1)
        image, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours to polygons + get bounding circles
        contours_poly = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

        # Find bounding circle for the marker (usually largest)
        centers = np.array(centers)
        radius = np.array(radius)
        if radius.size != 0:
            if radius.max() >= 20 and radius.max() < 200:
                max_index = np.where(radius == radius.max())
                print(max_index)
                max = int(max_index[0][0])

                self.detection_point[0] = int(centers[max][0])
                self.detection_point[1] = int(centers[max][1] + radius[max])
                self.dp_update = True

        # Draw polygonal contour, bounding circles, and detection point
        if centers.size != 0:
            if self.dp_update == True:
                cv2.drawContours(masked, contours_poly, max, self.bound_color)
                cv2.circle(masked, (int(centers[max][0]), int(centers[max][1])), int(radius[max]), self.bound_color, 1)
                cv2.circle(frame, (int(centers[max][0]), int(centers[max][1])), int(radius[max]), self.bound_color, 1)
                cv2.circle(frame, (self.detection_point[0], self.detection_point[1]), 5, self.center_color, -5)
            



    def main_segmentation(self):
        ''' Main Code '''
        while True:
            ret_val, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            start = time.time()
            mask = self.image_segmentation(frame)
            #mask = self.non_parametric_segmentation(frame)
            end = time.time()
            print(end - start)
            #self.blob_detection(frame, self.masked, mask)

            display = np.concatenate((frame, self.masked), axis=0)

            title = "Image Segmentation"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self.frame_width, int(self.frame_height*2))
            cv2.imshow(title, display)

            if cv2.waitKey(1) == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    start = segmentation()
    start.calibration()
    start.main_segmentation()
