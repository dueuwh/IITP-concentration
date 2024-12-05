import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        # Eyes with around skin segments
        # self.LEFT_EYE_POINTS = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
        # self.RIGHT_EYE_POINTS = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]
        
        # Only eyes
        # self.LEFT_EYE_POINTS = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
        # self.RIGHT_EYE_POINTS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        
        # Eyes with eyelid
        self.LEFT_EYE_POINTS = [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255]
        self.RIGHT_EYE_POINTS = [243, 190, 56, 28, 27, 29, 30, 247, 130, 25, 110, 24, 23, 22, 26, 112]
        
        self._analyze(original_frame, landmarks, side, calibration)
        self.blinking_state = 0
        

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = (p1.x + p2.x) / 2
        y = (p1.y + p2.y) / 2
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
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [region], (255, 255, 255))
        eye = cv2.bitwise_and(frame.copy(), frame.copy(), mask=mask)
        eye = cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY)
        
        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0])
        max_x = np.max(region[:, 0])
        min_y = np.min(region[:, 1])
        max_y = np.max(region[:, 1])

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
        top = self._middle_point(landmarks.multi_face_landmarks[0].landmark[points[3]], landmarks.multi_face_landmarks[0].landmark[points[5]])
        bottom = self._middle_point(landmarks.multi_face_landmarks[0].landmark[points[11]], landmarks.multi_face_landmarks[0].landmark[points[13]])

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
            self.blinking_state = True
        except ZeroDivisionError:
            ratio = None
            self.blinking_state = False
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
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            raise ValueError(f"size: {side} wrong value")

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
