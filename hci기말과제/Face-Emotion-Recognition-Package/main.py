from libs.FacialExpression import FacialExpressionRecognizer
import cv2
from libs.Face import FacialLandmarkDetector
import libs.Face as Face
import numpy as np

facial_landmark_detector = FacialLandmarkDetector(
        model=Face.FACIAL_LANDMARK_DETECTION_MODEL_DLIB,
        face_detector=Face.FACE_DETECTION_MODEL_OPENCV_DNN)

facial_expression_recognizer = FacialExpressionRecognizer()

facial_expression_recognizer.load("SVM_Test_c=1.pkl")

cap = cv2.VideoCapture(0) ##내장된 웹캠 =0번

image = cap.read() ##자동한 순간의 프레임 읽어오는 것
emotion = ""


def min_max_normalization(value):
    value = list(value)

    _max = max(value)
    _min = min(value)

    result = []

    for val in value:
        _val = (val - _min) / (_max - _min)
        result.append(_val)

    return np.array(result)

while True:##계속 반복

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)##0이 상하반전 1은 좌우반전

    if not ret:
        print("카메라 실패")
        break

    facial_landmark_detector.feed(frame)

    if facial_landmark_detector.getIsDetect():

        landmarks = facial_landmark_detector.getFacialLandmark()

        x = landmarks.getX()
        y = landmarks.getY()
        x = min_max_normalization(x)
        y = min_max_normalization(y)

        for i in range(68):
            cv2.circle(frame, (int(x[i]), int(y[i])), 1, (255, 255, 0), 3)

        features = []
        features.append(y[41]-y[36])
        features.append(y[46]-y[45])
        features.append(y[51])
        features.append(y[39] - y[21])
        features.append(y[42] - y[22])
        features.append(x[48])
        features.append(x[54])
        features.append(-y[44]+y[46])
        features.append(-y[37]+y[41])
        features.append(x[54]-x[48])
        features.append(-y[18])
        features.append(-y[25])
        features.append(y[57] - y[48])
        features.append(y[57] - y[54])
        features.append(-y[21])
        features.append(-y[22])

        facial_expression_recognizer.feed([features])

        result = facial_expression_recognizer.getPrediction()

        emotion = "Neutral"

        if result == 0:
            emotion = "anger"

        elif result == 1:
            emotion = "disgust"

        elif result == 2:
            emotion = "fear"

        elif result == 3:
            emotion = "happiness"

        elif result == 4:
            emotion = "neutral"

        elif result == 5:
            emotion = "sadness"

        elif result == 6:
            emotion = "surprise"

    # emotion = ""

    cv2.putText(frame, emotion, (10, 30), 1, 2, (0, 0, 255), 2) ##00255는 빨강, (10,30을 org로 저장)

    cv2.imshow("frame", frame) ##1000이 1초

    key = cv2.waitKey(30)

    if key == 27:
        break ##끄는 버튼(esc), ord(x)는 x로 끔 대소문자 구분됨
