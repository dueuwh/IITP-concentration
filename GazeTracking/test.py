import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

image_path = "C:/Users/U/Desktop/BCML/IITP/eye/data/new_image_eye.jpg"
image_load = cv2.imread(image_path)
image = cv2.cvtColor(image_load, cv2.COLOR_RGB2GRAY)

kernel = np.ones((3,3), np.uint8)
new_frame = cv2.bilateralFilter(image, 10, 15, 15)
new_frame = cv2.erode(new_frame, kernel, iterations=3)
new_frame = cv2.threshold(new_frame,50, 255, cv2.THRESH_BINARY)[1]

new_frame = np.invert(new_frame)

x = np.sum(new_frame, 0)
y = np.sum(new_frame, 1)

plt.imshow(new_frame)
plt.show()
plt.title("x")
plt.bar(np.linspace(0, len(x), len(x)), x)
plt.show()
plt.title("y")
plt.bar(np.linspace(0, len(y), len(y)), y)
plt.show()


