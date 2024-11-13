"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np

video_num = input("0 for IITP_눈_돌리기.mp4\n1 for IITP_고개_돌리기.mp4\n2  for webcam with rolling_eye label\nany keys for without label")
video_select = video_num

gaze = GazeTracking()
rolling_eye = "../data/IITP_눈_돌리기.mp4"
label4rolling_eye = [1 for i in range(30*30)] + [0 for i in range(3550)]
rolling_head = "../data/IITP_고개_돌리기.mp4"
label4rolling_head = [1 for i in range(30*29)] + [0 for i in range(3564)]
camera = 0

if video_select == 0:
    webcam = cv2.VideoCapture(rolling_eye)
    label = label4rolling_eye
elif video_select == 1:
    webcam = cv2.VideoCapture(rolling_head)
    label = label4rolling_head
elif video_select == 2:
    webcam = cv2.VideoCapture(0)
    label = label4rolling_eye
else:
    webcam = cv2.VideoCapture(0)
    label = False
fps = 30

temp_list = []

processed_frame_count = 0

sec30_accuracy = 0
min1_accuracy = 0
min3_accuracy = 0

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    # elif gaze.is_right() and not gaze.is_bottom() and not gaze.is_upper():
    #     text = "Looking right"
    # elif gaze.is_right() and gaze.is_bottom() and not gaze.is_upper():
    #     text = "Looking bottom right"
    # elif gaze.is_right() and not gaze.is_bottom() and gaze.is_upper():
    #     text = "Looking upper right"
    # elif gaze.is_left() and not gaze.is_bottom() and not gaze.is_upper():
    #     text = "Looking left"
    # elif gaze.is_center() and gaze.is_bottom() and not gaze.is_upper():
    #     text = "Looking bottom left"
    # elif gaze.is_bottom() and not gaze.is_bottom() and gaze.is_upper():
    #     text = "Looking upper left"
    # elif gaze.is_upper() and not gaze.is_right() and not gaze.is_left():
    #     text = "Looking upper"
    # elif gaze.is_bottom() and not gaze.is_right() and not gaze.is_left():
    #     text = "Looking bottom"
    elif gaze.is_center():
        text = "Focus"
    else:
        text = "Distracted"
    
    if text == "Focus":
        temp_list.append(1)
    else:
        temp_list.append(0)
    
    if len(temp_list) >= 180*fps:
        temp_list = temp_list[1:]
    
    if label:
        if processed_frame_count >= 30*fps:
            temp_pred = temp_list[-30*fps:]
            temp_label = label[processed_frame_count-(30*fps):processed_frame_count]
            temp_total_check = [1 for i in range(30*fps) if temp_pred[i] == temp_label[i]]
            sec30_accuracy = round(sum(temp_total_check)/(30*fps), 2)*100
        else:
            sec30_accuracy = None
        
        if processed_frame_count >= 60*fps:
            temp_pred = temp_list[-(60*fps):]
            temp_label = label[processed_frame_count-(60*fps):processed_frame_count]
            temp_total_check = [1 for i in range((60*fps)) if temp_pred[i] == temp_label[i]]
            min1_accuracy = round(sum(temp_total_check)/(60*fps), 2)*100
        else:
            min1_accuracy = None
        
        if processed_frame_count >= 180*fps:
            temp_pred = temp_list[-180*fps:]
            temp_label = label[processed_frame_count-180*fps:processed_frame_count]
            temp_total_check = [1 for i in range(180*fps) if temp_pred[i] == temp_label[i]]
            min3_accuracy = round(sum(temp_total_check)/(180*fps), 2)*100
        else:
            min3_accuracy = None
    else:
        sec30_accuracy = "No label"
        min1_accuracy = "No label"
        min3_accuracy = "No label"
    
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (147, 58, 31), 2)
    cv2.putText(frame, f"30 seconds concentration score: {round(sum(temp_list[-30*fps:])/(30*fps), 2)} Accuracy: {sec30_accuracy}%",
                (90, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)
    cv2.putText(frame, f"1 minute concentration score: {round(sum(temp_list[-60*fps:])/(60*fps), 2)} Accuracy: {min1_accuracy}%",
                (90, 160), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)
    cv2.putText(frame, f"3 minute concentration score: {round(sum(temp_list[-180*fps:])/(180*fps), 2)} Accuracy: {min3_accuracy}%",
                (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 255), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)

    cv2.imshow(f"IITP estimation of concentration", frame)

    processed_frame_count += 1
    
    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
