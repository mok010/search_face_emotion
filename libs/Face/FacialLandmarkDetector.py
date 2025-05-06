import cv2
import numpy as np
import math
from .FaceDetector import *
from libs.utils.Point import Point

FACIAL_LANDMARK_DETECTION_MODEL_DLIB = 0

'''
1adrianb's Facial Landmark Detector (Face Alignment)
https://github.com/1adrianb/face-alignment
'''


class FacialLandmarkDetector:
    def __init__(self, model=FACIAL_LANDMARK_DETECTION_MODEL_DLIB, face_detector=FACE_DETECTION_MODEL_DLIB):
        self.__detector__ = None
        self.__model_name__ = model
        self.__landmarks__ = None
        self.__face_detector__ = None
        self.__pre_roi__ = None
        self.__eye_landmark__ = None
        self.roi_moved = False

        if model == FACIAL_LANDMARK_DETECTION_MODEL_DLIB:
            import dlib
            self.__detector__ = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

            if face_detector == FACE_DETECTION_MODEL_SFD:
                print("Can't use SFD Face Detection Model with Dlib landmark detector")
                return

            self.__face_detector__ = FaceDetector(model=face_detector)

        else:
            print("No Facial Landmark Detection model")
            return

    def feed(self, img, tracking=False, dist=5):

        if self.__model_name__ == FACIAL_LANDMARK_DETECTION_MODEL_DLIB:
            self.landmark_detection_dlib(img)

    def feed_with_roi(self, img, roi):

        if self.__model_name__ == FACIAL_LANDMARK_DETECTION_MODEL_DLIB:
            self.landmark_detection_dlib(img, roi)

    def landmark_detection_dlib(self, img, face_roi=None):
        roi = None
        # feed image to face detection
        if face_roi is None:
            self.__face_detector__.feed(img)

            if self.__face_detector__.getIsDetect():
                # get face roi
                roi = self.__face_detector__.getFace()

            else:
                self.__landmarks__ = None
                self.__face_detector__.__is_detect__ = False

        else:
            roi = Face(x=face_roi[0], y=face_roi[1], w=face_roi[2], h=face_roi[3])

        if roi is not None:

            # dlib facial landmark detection
            import dlib
            dlib_roi = dlib.rectangle(roi.getX(), roi.getY(), roi.getX() + roi.getW(), roi.getY() + roi.getH())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            landmarks = self.__detector__(gray, dlib_roi)
            landmarks = self.shape_to_np(shape=landmarks)

            x = landmarks[:, 0]
            y = landmarks[:, 1]

            self.__landmarks__ = FacialLandmark(x=x, y=y)

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i][0] = shape.part(i).x
            coords[i][1] = shape.part(i).y

        # return the list of (x, y)-coordinates
        return coords

    def getIsDetect(self):
        return self.__face_detector__.getIsDetect()

    def getFace(self):
        return self.__face_detector__.getFace()

    def getFacialLandmark(self):
        return self.__landmarks__


class FacialLandmark:
    def __init__(self, x, y, z=None):
        self.size = len(x)
        self.x = x
        self.y = y
        self.z = z

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

