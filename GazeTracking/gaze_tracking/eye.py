import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
    RIGHT_EYE_POINTS = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(int(landmarks.multi_face_landmarks[0].landmark[point].x * frame.shape[1]), 
                            int(landmarks.multi_face_landmarks[0].landmark[point].y * frame.shape[0])) 
                           for point in points])
        
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (mediapipe object): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.multi_face_landmarks[0].landmark[points[0]].x, landmarks.multi_face_landmarks[0].landmark[points[0]].y)
        right = (landmarks.multi_face_landmarks[0].landmark[points[8]].x, landmarks.multi_face_landmarks[0].landmark[points[8]].y)
        top = self._middle_point(landmarks.multi_face_landmarks[0].landmark[points[3]].x, landmarks.multi_face_landmarks[0].landmark[points[5]].y)
        bottom = self._middle_point(landmarks.multi_face_landmarks[0].landmark[points[12]].x, landmarks.multi_face_landmarks[0].landmark[points[13]].y)

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (mediapipe object): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = LEFT_EYE_POINTS
        elif side == 1:
            points = RIGHT_EYE_POINTS
        else:
            raise ValueError(f"size: {side} wrong value")

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)