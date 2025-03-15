from __future__ import division
import os
import cv2
# import dlib
import mediapipe as mp
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        
        self.upper_border = 0.30
        self.bottom_border = 0.70
        self.right_border = 0.70
        self.left_border = 0.30
        self.blinking_ratio = 1.5
        
        self.landmark_state = 0
        
        # 0: self._predictor.process passed and landmarks.multi_face_landmarks is True
        # 1: self._predictor.process passed but landmarks.multi_face_landmarks is False
        # 2: self._predictor.process cause Error

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self._predictor = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame = self.frame
        # try:
        landmarks = self._predictor.process(frame)
        if landmarks.multi_face_landmarks:    
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)
            self.landmark_state = 1000
        else:
            self.eye_left = None
            self.eye_right = None
            self.landmark_state = 1001
        # except:
        #     self.eye_left = None
        #     self.eye_right = None
        #     self.landmark_state = 1002

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = (self.eye_left.pupil.x - self.eye_left.eye_h_border[0][0]) / self.eye_left.eye_h_length
            pupil_right = (self.eye_right.pupil.x - self.eye_right.eye_h_border[0][0]) / self.eye_right.eye_h_length
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = (self.eye_left.pupil.y - self.eye_left.eye_v_border[0][1]) / self.eye_left.eye_v_length
            pupil_right = (self.eye_right.pupil.y - self.eye_right.eye_v_border[0][1]) / self.eye_right.eye_v_length
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() >= self.right_border

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() <= self.left_border

    def is_bottom(self):
        """Returns true if the user is looking to the bottom"""
        if self.pupils_located:
            return self.vertical_ratio() > self.bottom_border
    
    def is_upper(self):
        """Returns true if the user is looking to the upper"""
        if self.pupils_located:
            return self.vertical_ratio() <= self.upper_border

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True and self.is_bottom() is not True and self.is_upper() is not True


    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > self.blinking_ratio

    # not used
    def get_direction(self):
        direction = {'upper':[False, 0], 'bottom':[False, 0], 'right':[False, 0], 'left':[False, 0]}
        if self.is_right():
            direction['right'] = [True, self.horizontal_ratio() - self.right_border]
        if self.is_left():
            direction['left'] = [True, self.horizontal_ratio() - self.left_border]
        if self.is_upper():
            direction['upper'] = [True, self.vertical_ratio() - self.upper_border]
        if self.is_right():
            direction['bottom'] = [True, self.vertical_ratio() - self.bottom_border]
        
        for key in direction.keys():
            if direction[key][0]:
                pass

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        
        if self.eye_left is None or self.eye_right is None:
            return frame, None, None
        else:
            return frame, self.eye_left.pupil.iris_frame.shape, self.eye_right.pupil.iris_frame.shape
