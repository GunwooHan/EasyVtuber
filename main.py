import time

import pyvirtualcam
import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
        print(f'Using virtual camera: {cam.device}')
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=None, fx=720/480, fy=720/480)
            results = holistic.process(frame)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            pad_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            pad_frame[:, (1280-960)//2:(1280-960)//2+960] = frame
            cam.send(pad_frame)
            cam.sleep_until_next_frame()