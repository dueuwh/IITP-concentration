import cv2
import numpy as np

# # Create a blank image with a white rectangle
# image = np.zeros((300, 300, 3), dtype="uint8")
# cv2.rectangle(image, (50, 50), (250, 250), (255, 255, 255), -1)  # White rectangle

# # Create a circular mask
# mask = np.zeros((300, 300), dtype="uint8")
# cv2.circle(mask, (150, 150), 100, 255, -1)  # White circle on black background

# # Apply the mask using bitwise_and
# masked_image = cv2.bitwise_and(image, image, mask=mask)

# # Display the original image, mask, and masked image
# cv2.imshow("Original Image", image)
# cv2.imshow("Mask", mask)
# cv2.imshow("Masked Image", masked_image)

# # Wait for a key press and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rolling_eye = "../data/IITP_눈_돌리기.mp4"
rolling_head = "../data/IITP_고개_돌리기.mp4"
webcam = cv2.VideoCapture(rolling_head)

print(int(webcam.get(cv2.CAP_PROP_FRAME_COUNT)))
if None:
    print("False")
else:
    print("True")