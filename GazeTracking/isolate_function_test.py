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

# rolling_eye = "../data/IITP_눈_돌리기.mp4"
# rolling_head = "../data/IITP_고개_돌리기.mp4"
# webcam = cv2.VideoCapture(rolling_head)

# print(int(webcam.get(cv2.CAP_PROP_FRAME_COUNT)))
# if None:
#     print("False")
# else:
#     print("True")

# def test_func():
#     for i in range(100):
#         yield i
#         continue

# if __name__ == "__main__":
#     a = test_func()
#     print(a)

video_path = "C:/Users/U/Desktop/BCML/Drone/DJI_20240629161442_Ins2-P4P.MP4"
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, 11000)
ret, frame = cap.read()

x, y, w, h = 1180, 760, 17, 17
cropped_image = frame[y:y+h, x:x+w]

cv2.imshow("", cropped_image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

output_path = f"C:/Users/U/Desktop/BCML/Drone/test_video_11000.png"
cv2.imwrite(output_path, frame)

# for i in range(100):
#     ret, frame = cap.read()
#     output_path = f"C:/Users/U/Desktop/BCML/Drone/extract/test_video_image_{10000+i}.png"
#     cv2.imwrite(output_path, frame)