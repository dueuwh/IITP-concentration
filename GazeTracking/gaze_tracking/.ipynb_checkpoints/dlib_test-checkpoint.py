import dlib
import os
import cv2
import time
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


import matplotlib.pyplot as plt
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

cap = cv2.VideoCapture(0)
start_time = time.time()
while True:
    end_time = time.time()
    duration = end_time - start_time
    if duration > 10:
        break
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame)
    try:
        landmarks = predictor(frame, faces[0])
        print("landmarks.part(): ", landmarks.parts())
    except IndexError:
        plt.imshow(frame)
        plt.show()

cap.release()
cv2.destroyAllWindows()




