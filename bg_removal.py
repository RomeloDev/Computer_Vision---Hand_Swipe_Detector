import cv2
import cvzone
import cvzone.FPS
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentator = SelfiSegmentation()
previous_time = 0
current_time = 0
imgBg1 = cv2.imread('assets/bg1.jpg')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOut = segmentator.removeBG(img, imgBg1, 0.50)
    
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    
    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(imgStacked, f"FPS: {int(fps)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('img out', imgStacked)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()