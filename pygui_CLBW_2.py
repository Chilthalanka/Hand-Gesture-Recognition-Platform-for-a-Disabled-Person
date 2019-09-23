import sys
import os
import resources_mainwindow
import resources_subwindow
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
#-----------------------------------------------
# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time

# global variables
bg = None
tic = None
toc = None
avg = None
preVal = None
avgHandVal = None


#--------------------------------------------------------------------------------

def Refresh():
    print("Working")

#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=30):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-------------------------------------------------------------------------------
# Function - To count the number of fingers in the segmented hand region
#-------------------------------------------------------------------------------
def count(thresholded, segmented):
    global avgXCenter, avgYCenter, avgRadius, cnt
    
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = (extreme_left[0] + extreme_right[0]) / 2
    cY = (extreme_top[1] + extreme_bottom[1]) / 2
    
    if (cnt <5):
        avgXCenter = (avgXCenter + cX)/5
        avgYCenter = (avgYCenter + cX)/5
    else:
        cnt = 0
        avgXCenter = cX
        avgYCenter = cY

    cnt +=1
    
    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]
	
    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.6 * maximum_distance)
	
    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (int(round(cX)), int(round(cY))), int(round(radius)), 255, 1)	
    Handcut = thresholded.copy()

	
    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    cv2.circle(Handcut, (int(round(cX)), int(round(cY))), int(round(radius)), 255, 1)
    cv2.imshow("Thresholded", Handcut)

    # compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1
    return count




#-------------------------------------------------------------------------------
# Function - To read camera
#-------------------------------------------------------------------------------
def camRead(_top, _right, _bottom, _left):
    global camera, height, width, roi, gray, clone

    # get the current frame 
    (grabbed, frame) = camera.read()

    # resize the frame
    frame = imutils.resize(frame, width=700)

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[_top:_bottom, _right:_left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    




#-------------------------------------------------------------------------------
# Function - To calibrate weighted average model
#-------------------------------------------------------------------------------
def calibWeightedModel(_num_frames):
    global accumWeight, gray    
    
    while (_num_frames < 40):        
        
        camRead(10, 350, 225, 590)  # read camera

        run_avg(gray, accumWeight)
        if _num_frames == 1:
            print("[STATUS] please wait! calibrating...")
        elif _num_frames == 39:
            print("[STATUS] calibration successful...")
        
        # increment the number of frames
        _num_frames += 1

    return _num_frames



#-------------------------------------------------------------------------------
# Function - To get the hand value average
#-------------------------------------------------------------------------------
def handValAvg():
    global fingers, firstFrame, tic, toc, avg, preVal, widget

    currVal = 0    
        
    if (firstFrame):
        tic = time.time()
        avg = fingers        
        firstFrame = False
    else:
        toc = time.time()
        if ((toc-tic) > 2):
            currVal = avg
            if ((currVal-preVal)==0):
                #print(int(avg))
                firstFrame = True
                try:
                    del widget.Subwindow
                    print(123)                    

                except Exception:
                    pass
                
                return int(avg)
            else:
                firstFrame = True
                avg = None
                return avg
        else:           
            avg = (avg + fingers)/2

    '''
    ### to put in main function if necessary ###
            if (firstFrame):
                tic = time.time()
                avg = fingers
                fingAvgCnt = 1
                firstFrame = False
            else:
                if ((time.time()-tic) > 5):
                    #cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    print(int(avg))
                    firstFrame = True
                else:
                    #fingAvgCnt += 1
                    avg = (avg + fingers)/2
    '''






#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

def main():
    global accumWeight, camera, height, width, roi, gray
    global fingers, firstFrame, clone
    global avgXCenter, avgYCenter, avgRadius, cnt
    global preVal
    global avgHandVal
    global widget 
    
    # Initialize parameters
    avgXCenter = 0
    avgYCenter = 0 
    cnt=0
        
    # Initialize Camera
    accumWeight = 0.5    
    camera = cv2.VideoCapture(0) # get the reference to the webcam    
    top, right, bottom, left = 10, 350, 225, 590  # region of interest (ROI) coordinates    
    num_frames = 0  # initialize num of frames    
    
    # calibrate the weighted model -> returns current num_frames
    num_frames = calibWeightedModel(num_frames)     

        
    firstFrame = True  
    
    while True:        
        camRead(10, 350, 225, 590)
        
        hand = segment(gray) # segment the hand region
        
        # check whether hand region is segmented
        if hand is not None:
            global avgHandVal
            # if yes, unpack the thresholded image and segmented region            
            (thresholded, segmented) = hand            
                
            # draw the segmented region and display the frame
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

            # count the number of fingers
            
            fingers = count(thresholded, segmented)
            cv2.putText(clone, str(fingers), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)            

            preVal = fingers
            
            avgHandVal = handValAvg()  # computes the average value shown from the hand for 5 seconds
            if(avgHandVal):
                exeCmd(avgHandVal)
                
                        
            # show the thresholded image
            #cv2.imshow("Thresholded", thresholded)
            
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
                break    


    # free up memory
    camera.release()
    cv2.destroyAllWindows()
    del widget
    exit()
    

class subwindow(QMainWindow):
    def __init__(self):
        super(QMainWindow,self).__init__()
        loadUi('subwindow.ui', self)
        self.show()
        
class mainwindow(QMainWindow):
    def __init__(self):
        super(QMainWindow,self).__init__()
        loadUi('mainwindow.ui', self)
        self.show()

        self.drink.clicked.connect(self.drink_clicked)
        self.eat.clicked.connect(self.eat_clicked)
        self.lavatory.clicked.connect(self.lava_clicked)
        self.sos.clicked.connect(self.sos_clicked)
    
    @pyqtSlot()
    def drink_clicked(self):
        self.Subwindow = subwindow()
        self.Subwindow.stackedWidget.setCurrentIndex(0)
        
    def eat_clicked(self):
        self.Subwindow = subwindow()
        self.Subwindow.stackedWidget.setCurrentIndex(1)

    def lava_clicked(self):
        self.Subwindow = subwindow()
        self.Subwindow.stackedWidget.setCurrentIndex(2)

    def sos_clicked(self):
        self.Subwindow = subwindow()
        self.Subwindow.stackedWidget.setCurrentIndex(4)

#---------------------------------------------------------------------------------
def exeCmd(avgHandVal):    
    if(avgHandVal):
        print(avgHandVal)

    if (avgHandVal==0 or avgHandVal==None):
        del widget.Subwindow

    if(avgHandVal==1):
        widget.drink.click()

    elif(avgHandVal==2):
        widget.eat.click()

    elif(avgHandVal==3):
        widget.lavatory.click()

    elif(avgHandVal==4):
        widget.sos.click()
    

#---------------------------------------------------------------------------------

app = QApplication(sys.argv)
widget = mainwindow()

main()
sys.exit(app.exec_())
