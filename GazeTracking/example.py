"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
import time
import matplotlib.pyplot as plt

video_select = int(input("0 for IITP_눈_돌리기.mp4\n1 for IITP_고개_돌리기.mp4\n2  for webcam with rolling_eye label\n3 for 3min experiment\n4 for EMMA_video_2.mp4\nany keys for without label\ninput: "))

gaze = GazeTracking()
rolling_eye = "../data/IITP_눈_돌리기.mp4"
label4rolling_eye = [1 for i in range(30*30)] + [0 for i in range(3550)]
rolling_head = "../data/IITP_고개_돌리기.mp4"
label4rolling_head = [1 for i in range(30*29)] + [0 for i in range(3564)]
exp_tutorial = [1 for i in range(30*30)] + [0 for i in range(30*30*4)] + [ 1 for i in range(30*30)]
label4emma2 = [1 for i in range(30*30)] + [0 for i in range(int(30*30*3))] + [1 for i in range(397)]
camera = 0

if video_select == 0:
    print("\nstart rolling eye\n")
    webcam = cv2.VideoCapture(rolling_eye)
    label = label4rolling_eye
elif video_select == 1:
    print("\nstart rolling head\n")
    webcam = cv2.VideoCapture(rolling_head)
    label = label4rolling_head
elif video_select == 2:
    print("\nstart webcam with rolling eye label\n")
    webcam = cv2.VideoCapture(0)
    label = label4rolling_eye
elif video_select == 3:
    print("start webcam with 3min test\nFirst 30sec: looking center\nSecond 30sec: Looking left\nThird 30sec: Looking right\nFourth 30sec: Looking bottom\nFifth 30sec: Looking Upper\nLast 30sec: Looking center")
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    label = exp_tutorial
elif video_select == 4:
    print("start EMMA_video_2.mp4 test")
    webcam = cv2.VideoCapture("../data/EMMA_video_2.mp4")
    label = label4emma2
else:
    print("\nstart webcam without label\n")
    webcam = cv2.VideoCapture(0)
    label = False
fps = 30

temp_list = []
session_index = []

processed_frame_count = 0

sec30_accuracy = 0
min1_accuracy = 0
min3_accuracy = 0
start_time = time.time()

while True:
    if video_select == 3:
        if time.time() - start_time >= 180:
            break
    if video_select == 2:
        if processed_frame_count == 30*29+2550:
            break
    # We get a new frame from the webcam
    ret, frame = webcam.read()

    if not ret:
        break
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame, left_eye_size, right_eye_size = gaze.annotated_frame()
    
    text = ""

    if gaze.is_blinking():
        text = "Distracted"
    elif gaze.is_right() and not gaze.is_bottom() and not gaze.is_upper():
        text = "Distracted"
    elif gaze.is_right() and gaze.is_bottom() and not gaze.is_upper():
        text = "Distracted"
    elif gaze.is_right() and not gaze.is_bottom() and gaze.is_upper():
        text = "Distracted"
    elif gaze.is_left() and not gaze.is_bottom() and not gaze.is_upper():
        text = "Distracted"
    elif gaze.is_center() and gaze.is_bottom() and not gaze.is_upper():
        text = "Distracted"
    elif gaze.is_bottom() and not gaze.is_bottom() and gaze.is_upper():
        text = "Distracted"
    elif gaze.is_upper() and not gaze.is_right() and not gaze.is_left():
        text = "Distractedr"
    elif gaze.is_bottom() and not gaze.is_right() and not gaze.is_left():
        text = "Distracted"
    elif gaze.is_center():
        text = "Focus"
    else:
        if left_eye_size != None and right_eye_size != None:
            text = "Distracted"
        else:
            text = "No iris recognition"
    
    if text == "Focus":
        temp_list.append(1)
    elif text == "No iris recognition":
        temp_list.append(-1)
    else:
        temp_list.append(0)
    
    if time.time() - start_time <= 30:
        session_index.append(0)
    elif 30 < time.time() - start_time <= 150:
        session_index.append(1)
    else:
        session_index.append(0)
    
    # if len(temp_list) >= 180*fps:
    #     temp_list = temp_list[1:]
    
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
    
    elapsed_time = time.time() - start_time
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)
    cv2.putText(frame, f"30 seconds concentration score: {round(sum(temp_list[-30*fps:])/(30*fps), 2)} Accuracy: {sec30_accuracy}%",
                (90, 120), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
    cv2.putText(frame, f"1 minute concentration score: {round(sum(temp_list[-60*fps:])/(60*fps), 2)} Accuracy: {min1_accuracy}%",
                (90, 160), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
    cv2.putText(frame, f"3 minute concentration score: {round(sum(temp_list[-180*fps:])/(180*fps), 2)} Accuracy: {min3_accuracy}%",
                (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
    cv2.putText(frame, f"Elapsed time: {int(elapsed_time//3600)}:{int(elapsed_time%3600//60)}:{int(elapsed_time%60)}",
                (90, 300), cv2.FONT_HERSHEY_DUPLEX, 0.4, (147, 58, 31), 2)

    if right_eye_size is None or left_eye_size is None:
        pass
    else:
        if right_eye_size[0] < 30 or left_eye_size[0] < 30 or right_eye_size[1] < 40 or left_eye_size[1] < 40:
            cv2.putText(frame, "Bad environment",
                        (90, 350), cv2.FONT_HERSHEY_DUPLEX, 1.0, (30, 30, 190), 2)
            cv2.putText(frame, "The face is too far away from camera",
                        (90, 380), cv2.FONT_HERSHEY_DUPLEX, 0.6, (30, 30, 190), 2)
            cv2.putText(frame, "Or the camera resolution is low",
                        (90, 410), cv2.FONT_HERSHEY_DUPLEX, 0.6, (30, 30, 190), 2)
            cv2.putText(frame, f"Eye size Right:{right_eye_size}, Left: {left_eye_size}",
                        (90, 450), cv2.FONT_HERSHEY_DUPLEX, 0.4, (30, 30, 190), 2)
        else:
            cv2.putText(frame, "Good environment",
                        (90, 350), cv2.FONT_HERSHEY_DUPLEX, 1.0, (30, 190, 30), 2)
            cv2.putText(frame, f"Eye size Right:{right_eye_size}, Left: {left_eye_size}",
                        (90, 450), cv2.FONT_HERSHEY_DUPLEX, 0.4, (30, 190, 30), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 255), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    cv2.imshow(f"IITP estimation of concentration for 3 minutes", frame)

    processed_frame_count += 1
    
    if cv2.waitKey(1) == 27:
        break

if label:
    final_check = []
    for i in range(len(session_index)):
        if session_index[i] == 0:
            if temp_list[i] == 1:
                final_check.append(1)
            else:
                final_check.append(0)
        else:
            if temp_list[i] == 1:
                final_check.append(0)
            elif temp_list[i] == 0:
                final_check.append(1)
            else:
                final_check.append(0)
    final_accuracy = round(sum(final_check)/(len(session_index)), 2)*100
    print(f"{'='*60}\nFinal accuracy: {final_accuracy}%\n{'='*60}")
else:
    final_check = [1 for i in range(30*30*6) if temp_list[i] == label[i]]
    final_accuracy = round(sum(final_check)/(30*30*6), 2) * 100
    print(f"{'='*60}\nFinal accuracy: {final_accuracy}%\n{'='*60}")
webcam.release()
cv2.destroyAllWindows()
