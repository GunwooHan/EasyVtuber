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
from utils import preprocessing_image, postprocessing_image, get_distance

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_false')
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


def get_pose(facial_landmarks):
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

    # if args.debug:
    #     h, w, c = frame.shape
    #     cv2.line(frame, [int(iris_l_center[0] * w), int(iris_l_center[1] * h)],
    #              [int(iris_r_center[0] * w), int(iris_r_center[1] * h)], (0, 128, 255), 2)

    z_angle = np.arctan2(iris_l_center[1] - iris_r_center[1], iris_l_center[0] - iris_r_center[0])  # * 180 / np.pi
    y_angle = np.arctan2(iris_l_center[2] - iris_r_center[2], iris_l_center[0] - iris_r_center[0])  # * 180 / np.pi

    iris_middle_center = np.mean(np.array([iris_l_center, iris_r_center]), axis=0)
    mouth_center = [0, 0, 0]

    for idx in list(map(lambda x: x[0], FACEMESH_LIPS)):
        pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
        mouth_center[0] += pos_x
        mouth_center[1] += pos_y
        mouth_center[2] += pos_z

    mouth_center = np.array(mouth_center) / len(FACEMESH_LIPS)
    mouth_h = facial_landmarks[13].y - facial_landmarks[14].y
    mouth_w = facial_landmarks[78].x - (facial_landmarks[409].x + facial_landmarks[375].x) / 2
    mouth_ratio = mouth_h / mouth_w

    x_angle = np.arctan2(iris_middle_center[2] - mouth_center[2], iris_middle_center[1] - mouth_center[1])  # * 180 / np.pi
    x_angle = np.arctan2(iris_middle_center[1] - mouth_center[1], iris_middle_center[2] - mouth_center[2])  # * 180 / np.pi

    # if args.debug:
    #     h, w, c = frame.shape
    #     cv2.line(frame, [int(iris_middle_center[0] * w), int(iris_middle_center[1] * h)],
    #              [int(mouth_center[0] * w), int(mouth_center[1] * h)], (0, 128, 255), 2)
    #     cv2.putText(frame, f'z angle:{z_angle * 180 / np.pi}',
    #                 [int(iris_middle_center[0] * w), int(iris_middle_center[1] * h) + 30], cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (255, 0, 0), 1)

    iris_rotation_l_h = get_distance(facial_landmarks[386], facial_landmarks[374])
    iris_rotation_l_w = get_distance(facial_landmarks[382], facial_landmarks[263])
    iris_rotation_r_h = get_distance(facial_landmarks[159], facial_landmarks[145])
    iris_rotation_r_w = get_distance(facial_landmarks[33], facial_landmarks[155])

    iris_rotation_l_h_temp = np.sqrt(
        (iris_l_center[0] - facial_landmarks[386].x) ** 2 + (iris_l_center[1] - facial_landmarks[386].y) ** 2)
    iris_rotation_l_w_temp = np.sqrt(
        (iris_l_center[0] - facial_landmarks[382].x) ** 2 + (iris_l_center[1] - facial_landmarks[382].y) ** 2)
    iris_rotation_r_h_temp = np.sqrt(
        (iris_r_center[0] - facial_landmarks[159].x) ** 2 + (iris_r_center[1] - facial_landmarks[159].y) ** 2)
    iris_rotation_r_w_temp = np.sqrt(
        (iris_r_center[0] - facial_landmarks[33].x) ** 2 + (iris_r_center[1] - facial_landmarks[33].y) ** 2)

    # if args.debug:
    #     h, w, c = frame.shape
    #     cv2.line(frame, [int(iris_l_center[0] * w), int(iris_l_center[1] * h)],
    #              [int(iris_l_center[0] * w), int(iris_l_center[1] * h)], (0, 128, 128), 3)
    #     cv2.line(frame, [int(iris_r_center[0] * w), int(iris_r_center[1] * h)],
    #              [int(iris_r_center[0] * w), int(iris_r_center[1] * h)], (255, 128, 0), 3)

    eye_x_ratio = ((iris_rotation_l_w_temp / iris_rotation_l_w + iris_rotation_r_w_temp / iris_rotation_r_w) - 1) * 3
    eye_y_ratio = ((iris_rotation_l_h_temp / iris_rotation_l_h + iris_rotation_r_h_temp / iris_rotation_r_h) - 1) * 3

    eye_l_h_temp = 1 - 2 * (facial_landmarks[145].y - facial_landmarks[159].y) / (
                facial_landmarks[155].x - facial_landmarks[33].x)
    eye_r_h_temp = 1 - 2 * (facial_landmarks[374].y - facial_landmarks[386].y) / (
                facial_landmarks[263].x - facial_landmarks[382].x)

    return eye_l_h_temp, eye_r_h_temp, mouth_ratio, eye_y_ratio, eye_x_ratio, y_angle, z_angle, x_angle

def main():
    model = TalkingAnimeLight().cuda()
    model = model.eval()
    model = model.half()
    img = Image.open("character/sleeping_rat.png")
    img = img.resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0)

    mp_facemesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)

    with mp_facemesh.FaceMesh(refine_landmarks=True) as facemesh:

        ret, frame = cap.read()

        if ret is None:
            raise Exception("Can't find Camera")

        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = facemesh.process(input_frame)

        if results.multi_face_landmarks is None:
            raise Exception("Fail to initailize parameters")

        facial_landmarks = results.multi_face_landmarks[0].landmark

        mouth_eye_vector = torch.empty(1, 27)
        pose_vector = torch.empty(1, 3)

        input_image = input_image.half()
        mouth_eye_vector = mouth_eye_vector.half()
        pose_vector = pose_vector.half()

        input_image = input_image.cuda()
        mouth_eye_vector = mouth_eye_vector.cuda()
        pose_vector = pose_vector.cuda()

        pose_queue = []

        while cap.isOpened():
            ret, frame = cap.read()
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = facemesh.process(input_frame)

            if results.multi_face_landmarks is None:
                continue

            facial_landmarks = results.multi_face_landmarks[0].landmark
            #
            # iris_r_center = [0, 0, 0]
            # iris_l_center = [0, 0, 0]
            #
            # for idx in list(map(lambda x: x[0], FACEMESH_RIGHT_IRIS)):
            #     pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
            #     iris_r_center[0] += pos_x
            #     iris_r_center[1] += pos_y
            #     iris_r_center[2] += pos_z
            #
            # iris_r_center = np.array(iris_r_center) / len(FACEMESH_RIGHT_IRIS)
            #
            # for idx in list(map(lambda x: x[0], FACEMESH_LEFT_IRIS)):
            #     pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
            #     iris_l_center[0] += pos_x
            #     iris_l_center[1] += pos_y
            #     iris_l_center[2] += pos_z
            #
            # iris_l_center = np.array(iris_l_center) / len(FACEMESH_LEFT_IRIS)
            #
            # if args.debug:
            #     h, w, c = frame.shape
            #     cv2.line(frame, [int(iris_l_center[0] * w), int(iris_l_center[1] * h)],
            #              [int(iris_r_center[0] * w), int(iris_r_center[1] * h)], (0, 128, 255), 2)
            #
            # z_angle = np.arctan2(iris_l_center[1] - iris_r_center[1],
            #                      iris_l_center[0] - iris_r_center[0])  # * 180 / np.pi
            # y_angle = np.arctan2(iris_l_center[2] - iris_r_center[2],
            #                      iris_l_center[0] - iris_r_center[0])  # * 180 / np.pi
            #
            # iris_middle_center = np.mean(np.array([iris_l_center, iris_r_center]), axis=0)
            # mouth_center = [0, 0, 0]
            #
            # for idx in list(map(lambda x: x[0], FACEMESH_LIPS)):
            #     pos_x, pos_y, pos_z = facial_landmarks[idx].x, facial_landmarks[idx].y, facial_landmarks[idx].z
            #     mouth_center[0] += pos_x
            #     mouth_center[1] += pos_y
            #     mouth_center[2] += pos_z
            #
            # mouth_center = np.array(mouth_center) / len(FACEMESH_LIPS)
            # mouth_h = facial_landmarks[13].y - facial_landmarks[14].y
            # mouth_w = facial_landmarks[78].x - (facial_landmarks[409].x + facial_landmarks[375].x) / 2
            # mouth_ratio = mouth_h / mouth_w
            #
            # if args.debug:
            #     h, w, c = frame.shape
            #     cv2.line(frame, [int(iris_middle_center[0] * w), int(iris_middle_center[1] * h)],
            #              [int(mouth_center[0] * w), int(mouth_center[1] * h)], (0, 128, 255), 2)
            #     cv2.putText(frame, f'z angle:{z_angle * 180 / np.pi}',
            #                 [int(iris_middle_center[0] * w), int(iris_middle_center[1] * h) + 30],
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            #
            # iris_rotation_l_h = get_distance(facial_landmarks[386], facial_landmarks[374])
            # iris_rotation_l_w = get_distance(facial_landmarks[382], facial_landmarks[263])
            # iris_rotation_r_h = get_distance(facial_landmarks[159], facial_landmarks[145])
            # iris_rotation_r_w = get_distance(facial_landmarks[33], facial_landmarks[155])
            #
            # iris_rotation_l_h_temp = np.sqrt(
            #     (iris_l_center[0] - facial_landmarks[386].x) ** 2 + (iris_l_center[1] - facial_landmarks[386].y) ** 2)
            # iris_rotation_l_w_temp = np.sqrt(
            #     (iris_l_center[0] - facial_landmarks[382].x) ** 2 + (iris_l_center[1] - facial_landmarks[382].y) ** 2)
            # iris_rotation_r_h_temp = np.sqrt(
            #     (iris_r_center[0] - facial_landmarks[159].x) ** 2 + (iris_r_center[1] - facial_landmarks[159].y) ** 2)
            # iris_rotation_r_w_temp = np.sqrt(
            #     (iris_r_center[0] - facial_landmarks[33].x) ** 2 + (iris_r_center[1] - facial_landmarks[33].y) ** 2)
            #
            # if args.debug:
            #     h, w, c = frame.shape
            #     cv2.line(frame, [int(iris_l_center[0] * w), int(iris_l_center[1] * h)],
            #              [int(iris_l_center[0] * w), int(iris_l_center[1] * h)], (0, 128, 128), 3)
            #     cv2.line(frame, [int(iris_r_center[0] * w), int(iris_r_center[1] * h)],
            #              [int(iris_r_center[0] * w), int(iris_r_center[1] * h)], (255, 128, 0), 3)
            #
            # eye_x_ratio = ((
            #                            iris_rotation_l_w_temp / iris_rotation_l_w + iris_rotation_r_w_temp / iris_rotation_r_w) - 1) * 3
            # eye_y_ratio = ((
            #                            iris_rotation_l_h_temp / iris_rotation_l_h + iris_rotation_r_h_temp / iris_rotation_r_h) - 1) * 3
            #
            # eye_l_h_temp = 0.9 - 2 * (facial_landmarks[145].y - facial_landmarks[159].y) / (
            #             facial_landmarks[155].x - facial_landmarks[33].x)
            # eye_r_h_temp = 0.9 - 2 * (facial_landmarks[374].y - facial_landmarks[386].y) / (
            #             facial_landmarks[263].x - facial_landmarks[382].x)

            pose = get_pose(facial_landmarks)

            if len(pose_queue) < 3:
                pose_queue.append(pose)
                pose_queue.append(pose)
                pose_queue.append(pose)
            else:
                pose_queue.pop(0)
                pose_queue.append(pose)

            np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.6, 0.3, 0.1])

            eye_l_h_temp = np_pose[0]
            eye_r_h_temp = np_pose[1]
            mouth_ratio = np_pose[2]
            eye_y_ratio = np_pose[3]
            eye_x_ratio = np_pose[4]
            y_angle = np_pose[5]
            z_angle = np_pose[6]
            x_angle = np_pose[7]

            mouth_eye_vector[0, :] = 0

            mouth_eye_vector[0, 2] = eye_l_h_temp
            mouth_eye_vector[0, 3] = eye_r_h_temp

            mouth_eye_vector[0, 14] = mouth_ratio * 1.5

            mouth_eye_vector[0, 25] = eye_y_ratio
            mouth_eye_vector[0, 26] = eye_x_ratio

            # pose_vector[0, 0] = x_angle
            pose_vector[0, 0] = (1.5 + x_angle) * 2.0
            pose_vector[0, 1] = y_angle * 2.0   #temp weight
            pose_vector[0, 2] = z_angle * 1.5   #temp weight

            cv2.putText(frame, f'z angle:{x_angle * 180 / np.pi}, {x_angle}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            output_image = model(input_image, mouth_eye_vector, pose_vector)
            cv2.imshow("frame", cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2BGRA))
            if args.debug:
                cv2.imshow("camera", frame)
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
