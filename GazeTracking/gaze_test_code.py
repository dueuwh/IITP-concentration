import os
import sys
sys.executable

import matplotlib.pyplot as plt
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
while True:
    
    ret, frame = webcam.read()
    
    gaze.refresh(frame)
    
    gaze
