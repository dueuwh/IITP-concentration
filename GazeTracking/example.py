"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
rolling_eye = "../data/IITP_눈_돌리기.mp4"
rolling_head = "../data/IITP_고개_돌리기.mp4"
camera = 0
webcam = cv2.VideoCapture(camera)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right() and not gaze.is_below() and not gaze.is_above():
        text = "Looking right"
    elif gaze.is_right() and gaze.is_below() and not gaze.is_above():
        text = "Looking bottom right"
    elif gaze.is_right() and not gaze.is_below() and gaze.is_above():
        text = "Looking upper right"
    elif gaze.is_left() and not gaze.is_below() and not gaze.is_above():
        text = "Looking left"
    elif gaze.is_center() and gaze.is_below() and not gaze.is_above():
        text = "Looking bottom left"
    elif gaze.is_below() and not gaze.is_below() and gaze.is_above():
        text = "Looking upper left"
    elif gaze.is_above() and not gaze.is_rigth() and not gaze.is_left():
        text = "Looking upper"
    elif gaze.is_below() and not gaze.is_right() and not gaze.is_left():
        text = "Looking bottom"
    elif gaze.is_center():
        text = "Looking center"
    else:
        text = "Distracted"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
