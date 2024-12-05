import sys
print(sys.executable)
import cv2
import numpy as np


# Load the image
image = cv2.imread('C:/Users/U/Pictures/Camera Roll/WIN_20241205_14_15_50_Pro.jpg')

# Define the region of interest (ROI) you want to crop
# For example, let's say you want to crop a rectangle from (x1, y1) to (x2, y2)
x1, y1 = 100, 100
x2, y2 = 300, 300

# Create a mask with the same dimensions as the image, initialized to zeros (black)
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Fill the ROI on the mask with white (255)
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

new_mask = np.zeros(image.shape[:2], dtype=np.uint8)

cv2.fillPoly(new_mask, [np.array([[0, 0], [100, 0], [100, 100], [0, 100]])], (255, 255, 255))

new_roi = cv2.bitwise_and(image, image, mask=new_mask)

# Use the inverted mask to black out the area outside the ROI in the original image
image_bg = cv2.bitwise_and(image, image, mask=mask_inv)

# Use the original mask to extract the ROI from the original image
roi = cv2.bitwise_and(image, image, mask=mask)

# Combine the two images to get the final result
result = cv2.add(image_bg, roi)

# Crop the ROI from the result
cropped_image = result[y1:y2, x1:x2]

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Inverted Mask', mask_inv)
cv2.imshow('Background', image_bg)
cv2.imshow('ROI', roi)
cv2.imshow("New ROI", new_roi)
cv2.imshow('Cropped Image', cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()