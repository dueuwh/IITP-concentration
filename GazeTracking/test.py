import os
import cv2
import gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    
    ret, frame = webcam.read()
    
    gaze.refresh(frame)
    
    frame = gaze.annotated_frame()
    