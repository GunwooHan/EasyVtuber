# import time
#
# import pyvirtualcam
# import numpy as np
# import cv2
# import mediapipe as mp
#
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic
#
# cap = cv2.VideoCapture(0)
#
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
#         print(f'Using virtual camera: {cam.device}')
#         while cap.isOpened():
#             ret, frame = cap.read()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, dsize=None, fx=720/480, fy=720/480)
#             results = holistic.process(frame)
#             mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
#             mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#             pad_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
#             pad_frame[:, (1280-960)//2:(1280-960)//2+960] = frame
#             cam.send(pad_frame)
#             cam.sleep_until_next_frame()

import time
import argparse

import cv2
import torch
import torch.nn as nn
import pyvirtualcam
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

from tha2.poser.modes.mode_20 import load_face_morpher, load_face_rotater, load_combiner
from utils import preprocessing_image, postprocessing_image

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--output', type=str, default='window')
args = parser.parse_args()


class TalkingAnimeLight(nn.Module):
    def __init__(self):
        super(TalkingAnimeLight, self).__init__()
        self.face_morpher = load_face_morpher('pretrained/face_morpher.pt')
        self.two_algo_face_rotator = load_face_rotater('pretrained/two_algo_face_rotator.pt')
        self.combiner = load_combiner('pretrained/combiner.pt')

    def forward(self, image, mouth_eye_vector, pose_vector):
        x = image.clone()
        mouth_eye_morp_image = self.face_morpher(image[:, :, 32:224, 32:224], mouth_eye_vector)
        x[:, :, 32:224, 32:224] = mouth_eye_morp_image
        rotate_image = self.two_algo_face_rotator(x, pose_vector)[:2]
        output_image = self.combiner(rotate_image[0], rotate_image[1], pose_vector)
        return output_image


def main():
    model = TalkingAnimeLight().cuda()
    model = model.eval()
    model = model.half()
    img = Image.open("character/sleeping_rat.png")
    input_image = preprocessing_image(img).unsqueeze(0)

    mp_facemesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)

    with mp_facemesh.FaceMesh(refine_landmarks=True) as facemesh:

        ret, frame = cap.read()

        if ret is None:
            raise Exception("Can't find Camera")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = facemesh.process(frame)

        if results.multi_face_landmarks is None:
            raise Exception("Fail to initailize parameters")

        facial_landmarks = results.multi_face_landmarks[0].landmark

        # initial parameters
        eye_l_h = facial_landmarks[145].y - facial_landmarks[159].y
        eye_r_h = facial_landmarks[374].y - facial_landmarks[386].y

        eye_l_initial = facial_landmarks[159].y
        eye_r_initial = facial_landmarks[386].y

        mouth_eye_vector = torch.empty(1, 27)
        pose_vector = torch.empty(1, 3)

        input_image = input_image.half()
        mouth_eye_vector = mouth_eye_vector.half()
        pose_vector = pose_vector.half()

        input_image = input_image.cuda()
        mouth_eye_vector = mouth_eye_vector.cuda()
        pose_vector = pose_vector.cuda()

        while cap.isOpened():
            ret, frame = cap.read()
            results = facemesh.process(frame)

            if results.multi_face_landmarks is None:
                continue

            facial_landmarks = results.multi_face_landmarks[0].landmark

            iris_r_center = [0, 0, 0]
            iris_l_center = [0, 0, 0]

            for idx in list(map(lambda x: x[0], FACEMESH_RIGHT_IRIS)):
                pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
                iris_r_center[0] += pos_x
                iris_r_center[1] += pos_y
                iris_r_center[2] += pos_z

            iris_r_center = np.array(iris_r_center) / len(FACEMESH_RIGHT_IRIS)

            for idx in list(map(lambda x: x[0], FACEMESH_LEFT_IRIS)):
                pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
                iris_l_center[0] += pos_x
                iris_l_center[1] += pos_y
                iris_l_center[2] += pos_z

            iris_l_center = np.array(iris_l_center) / len(FACEMESH_LEFT_IRIS)

            z_angle = np.arctan2(iris_l_center[1] - iris_r_center[1], iris_l_center[0] - iris_r_center[0]) #* 180 / np.pi
            y_angle = np.arctan2(iris_l_center[2] - iris_r_center[2], iris_l_center[0] - iris_r_center[0]) #* 180 / np.pi

            iris_middle_center = np.mean([iris_l_center, iris_r_center])
            mouth_center = [0, 0, 0]

            for idx in list(map(lambda x: x[0], FACEMESH_LIPS)):
                pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
                mouth_center[0] += pos_x
                mouth_center[1] += pos_y
                mouth_center[2] += pos_z

            mouth_center = np.array(mouth_center) / len(FACEMESH_LIPS)
            mouth_h = facial_landmarks[13].y - facial_landmarks[14].y
            mouth_w = facial_landmarks[78].x - (facial_landmarks[409].x + facial_landmarks[375].x)/2
            mouth_ratio = mouth_h / mouth_w

            eye_l_x_min_boundary = min(facial_landmarks[33].x, facial_landmarks[145].x, facial_landmarks[159].x, facial_landmarks[155].x)
            eye_l_x_max_boundary = max(facial_landmarks[33].x, facial_landmarks[145].x, facial_landmarks[159].x, facial_landmarks[155].x)
            eye_l_y_min_boundary = min(facial_landmarks[33].y, facial_landmarks[145].y, facial_landmarks[159].y, facial_landmarks[155].y)
            eye_l_y_max_boundary = max(facial_landmarks[33].y, facial_landmarks[145].y, facial_landmarks[159].y, facial_landmarks[155].y)

            eye_r_x_min_boundary = min(facial_landmarks[382].x, facial_landmarks[386].x, facial_landmarks[374].x, facial_landmarks[263].x)
            eye_r_x_max_boundary = max(facial_landmarks[382].x, facial_landmarks[386].x, facial_landmarks[374].x, facial_landmarks[263].x)
            eye_r_y_min_boundary = min(facial_landmarks[382].y, facial_landmarks[386].y, facial_landmarks[374].y, facial_landmarks[263].y)
            eye_r_y_max_boundary = max(facial_landmarks[382].y, facial_landmarks[386].y, facial_landmarks[374].y, facial_landmarks[263].y)

            eye_l_x_ratio = (iris_l_center[0] - eye_l_x_min_boundary) / (eye_l_x_max_boundary - eye_l_x_min_boundary)
            eye_l_y_ratio = (iris_l_center[1] - eye_l_y_min_boundary) / (eye_l_y_max_boundary - eye_l_y_min_boundary)

            eye_r_x_ratio = (iris_r_center[0] - eye_r_x_min_boundary) / (eye_r_x_max_boundary - eye_r_x_min_boundary)
            eye_r_y_ratio = (iris_r_center[1] - eye_r_y_min_boundary) / (eye_r_y_max_boundary - eye_r_y_min_boundary)

            eye_x_ratio = (eye_l_x_ratio + eye_r_x_ratio) / 2
            eye_y_ratio = (eye_l_y_ratio + eye_r_y_ratio) / 2

            eye_l_h_temp = (facial_landmarks[159].y - eye_l_initial) / eye_l_h
            eye_r_h_temp = (facial_landmarks[386].y - eye_r_initial) / eye_r_h

            # eye_l_h_temp = (facial_landmarks[145].x - facial_landmarks[159].x) / (facial_landmarks[155].x - facial_landmarks[33].x) * 5
            # eye_r_h_temp = (facial_landmarks[374].x - facial_landmarks[386].x) / (facial_landmarks[263].x - facial_landmarks[382].x) * 5

            mouth_eye_vector[0, :] = 0

            mouth_eye_vector[0, 2] = eye_l_h_temp * 1.2
            mouth_eye_vector[0, 3] = eye_r_h_temp * 1.2

            mouth_eye_vector[0, 14] = mouth_ratio * 1.5

            mouth_eye_vector[0, 25] = eye_x_ratio
            mouth_eye_vector[0, 26] = eye_y_ratio

            # pose_vector[0, 0] = x_angle
            pose_vector[0, 0] = 0
            pose_vector[0, 1] = y_angle * 2.0   #temp weight
            pose_vector[0, 2] = z_angle * 1.5   #temp weight

            output_image = model(input_image, mouth_eye_vector, pose_vector)
            cv2.imshow("frame", cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2BGRA))
            cv2.waitKey(1)

# if args.output == 'window':

# wink                          0, 1
# happy wink                    2, 3
# surprised                     4, 5
# relaxed                       6, 7
# unimpressed                   8, 9
# iris Shrinkage L              12
# iris Shrinkage R              13
# mouth aaa                     14
# mouth iii                     15
# mouth uuu                     16
# mouth eee                     17
# mouth ooo                     18
# mouth delta                   19
# mouth lowered corner          18,19
# mouth raised corner           20,21
# mouth smirk                   22
# Iris rotation                 25, 26

# x,y,z                         27, 28, 29

if __name__ == '__main__':
    main()
