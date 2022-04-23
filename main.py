import argparse

import cv2
import torch
import pyvirtualcam
import numpy as np
import mediapipe as mp
from PIL import Image

from models import TalkingAnimeLight
from pose import get_pose
from utils import preprocessing_image, postprocessing_image

import errno
import json
import os
import queue
import socket
import time
from multiprocessing import Value, Process, Queue

from tha2.mocap.ifacialmocap_constants import *
from tha2.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--input', type=str, default='cam')
parser.add_argument('--character', type=str, default='test1')
parser.add_argument('--output_dir', type=str, default=f'dst')
parser.add_argument('--output_webcam', action='store_true')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def create_default_blender_data():
    data = {}

    for blendshape_name in BLENDSHAPE_NAMES:
        data[blendshape_name] = 0.0

    data[HEAD_BONE_X] = 0.0
    data[HEAD_BONE_Y] = 0.0
    data[HEAD_BONE_Z] = 0.0
    data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[LEFT_EYE_BONE_X] = 0.0
    data[LEFT_EYE_BONE_Y] = 0.0
    data[LEFT_EYE_BONE_Z] = 0.0
    data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[RIGHT_EYE_BONE_X] = 0.0
    data[RIGHT_EYE_BONE_Y] = 0.0
    data[RIGHT_EYE_BONE_Z] = 0.0
    data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    return data


class ClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = "0.0.0.0"
        self.port = 50002
        self.perf_time = 0

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind((self.address, self.port))
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    continue
                else:
                    raise e
            socket_string = socket_bytes.decode("utf-8")
            blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(blender_data)
            cur_time = time.perf_counter()
            fps = 1 / (cur_time - self.perf_time)
            self.perf_time = cur_time
            print(fps)
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        shape_key_values = blender_data["shapeKeyValues"]
        for blendshape_name in BLENDSHAPE_NAMES:
            data[blendshape_name] = shape_key_values[blendshape_name]

        head_bone = blender_data["boneValues"]["armatures"]["bones"]["headBone"]
        data[HEAD_BONE_X] = head_bone["e_rx"]
        data[HEAD_BONE_Y] = head_bone["e_ry"]
        data[HEAD_BONE_Z] = head_bone["e_rz"]
        data[HEAD_BONE_QUAT] = [
            head_bone["q_rx"],
            head_bone["q_ry"],
            head_bone["q_rz"],
            head_bone["q_w"],
        ]

        right_eye_bone = blender_data["boneValues"]["armatures"]["bones"]["rightEyeBone"]
        data[RIGHT_EYE_BONE_X] = right_eye_bone["e_rx"]
        data[RIGHT_EYE_BONE_Y] = right_eye_bone["e_ry"]
        data[RIGHT_EYE_BONE_Z] = right_eye_bone["e_rz"]
        data[RIGHT_EYE_BONE_QUAT] = [
            right_eye_bone["q_rx"],
            right_eye_bone["q_ry"],
            right_eye_bone["q_rz"],
            right_eye_bone["q_w"],
        ]

        left_eye_bone = blender_data["boneValues"]["armatures"]["bones"]["leftEyeBone"]
        data[LEFT_EYE_BONE_X] = left_eye_bone["e_rx"]
        data[LEFT_EYE_BONE_Y] = left_eye_bone["e_ry"]
        data[LEFT_EYE_BONE_Z] = left_eye_bone["e_rz"]
        data[LEFT_EYE_BONE_QUAT] = [
            left_eye_bone["q_rx"],
            left_eye_bone["q_ry"],
            left_eye_bone["q_rz"],
            left_eye_bone["q_w"],
        ]

        return data


@torch.no_grad()
def main():
    model = TalkingAnimeLight().to(device)
    model = model.eval()
    model = model
    img = Image.open(f"character/{args.character}.png")
    img = img.resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0)

    client_process = ClientProcess()
    client_process.start()

    if args.input == 'cam':
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        if ret is None:
            raise Exception("Can't find Camera")
    else:
        cap = cv2.VideoCapture(args.input)
        frame_count = 0
        os.makedirs(os.path.join('dst', args.character, args.output_dir), exist_ok=True)

    facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    if args.output_webcam:
        cam = pyvirtualcam.Camera(width=1280, height=720, fps=30)
        print(f'Using virtual camera: {cam.device}')

    mouth_eye_vector = torch.empty(1, 27)
    pose_vector = torch.empty(1, 3)

    # input_image = input_image.half()
    # mouth_eye_vector = mouth_eye_vector.half()
    # pose_vector = pose_vector.half()

    input_image = input_image.to(device)
    mouth_eye_vector = mouth_eye_vector.to(device)
    pose_vector = pose_vector.to(device)

    pose_queue = []
    blender_data = create_default_blender_data()

    while True:
        # ret, frame = cap.read()
        # input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = facemesh.process(input_frame)
        tic = time.perf_counter()

        try:

            new_blender_data = blender_data
            while not client_process.should_terminate.value and not client_process.queue.empty():
                new_blender_data = client_process.queue.get_nowait()
            blender_data = new_blender_data
        except queue.Empty:
            pass

        ifacialmocap_pose = blender_data

        # if results.multi_face_landmarks is None:
        #     continue
        #
        # facial_landmarks = results.multi_face_landmarks[0].landmark
        #
        # if args.debug:
        #     pose, debug_image = get_pose(facial_landmarks, frame)
        # else:
        #     pose = get_pose(facial_landmarks)
        #
        # if len(pose_queue) < 3:
        #     pose_queue.append(pose)
        #     pose_queue.append(pose)
        #     pose_queue.append(pose)
        # else:
        #     pose_queue.pop(0)
        #     pose_queue.append(pose)
        #
        # np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.6, 0.3, 0.1])
        #
        # eye_l_h_temp = np_pose[0]
        # eye_r_h_temp = np_pose[1]
        # mouth_ratio = np_pose[2]
        # eye_y_ratio = np_pose[3]
        # eye_x_ratio = np_pose[4]
        # x_angle = np_pose[5]
        # y_angle = np_pose[6]
        # z_angle = np_pose[7]
        eye_l_h_temp = ifacialmocap_pose[EYE_BLINK_LEFT]
        eye_r_h_temp = ifacialmocap_pose[EYE_BLINK_RIGHT]
        mouth_ratio = (ifacialmocap_pose[JAW_OPEN] - 0.10)
        x_angle = -ifacialmocap_pose[HEAD_BONE_X] * 1.5 + 1.57
        y_angle = -ifacialmocap_pose[HEAD_BONE_Y]
        z_angle = ifacialmocap_pose[HEAD_BONE_Z] - 1.57

        eye_x_ratio = (ifacialmocap_pose[EYE_LOOK_IN_LEFT] -
                       ifacialmocap_pose[EYE_LOOK_OUT_LEFT] -
                       ifacialmocap_pose[EYE_LOOK_IN_RIGHT] +
                       ifacialmocap_pose[EYE_LOOK_OUT_RIGHT]) / 2.0 / 0.75

        eye_y_ratio = (ifacialmocap_pose[EYE_LOOK_UP_LEFT]
                       + ifacialmocap_pose[EYE_LOOK_UP_RIGHT]
                       - ifacialmocap_pose[EYE_LOOK_DOWN_RIGHT]
                       + ifacialmocap_pose[EYE_LOOK_DOWN_LEFT]) / 2.0 / 0.75

        # print(np_pose[2],(ifacialmocap_pose[JAW_OPEN] - 0.10) )

        mouth_eye_vector[0, :] = 0

        mouth_eye_vector[0, 2] = eye_l_h_temp
        mouth_eye_vector[0, 3] = eye_r_h_temp

        mouth_eye_vector[0, 14] = mouth_ratio * 1.5

        mouth_eye_vector[0, 25] = eye_y_ratio
        mouth_eye_vector[0, 26] = eye_x_ratio

        pose_vector[0, 0] = (x_angle - 1.5) * 1.6
        pose_vector[0, 1] = y_angle * 2.0  # temp weight
        pose_vector[0, 2] = (z_angle + 1.5) * 2  # temp weight

        output_image = model(input_image, mouth_eye_vector, pose_vector)

        if args.debug:
            output_frame = cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2BGR)
            # resized_frame = cv2.resize(output_frame, (np.min(debug_image.shape[:2]), np.min(debug_image.shape[:2])))
            # output_frame = np.concatenate([debug_image, resized_frame], axis=1)
            cv2.imshow("frame", output_frame)
            # cv2.imshow("camera", debug_image)
            cv2.waitKey(1)
        if args.input != 'cam':
            cv2.imwrite(os.path.join('dst', args.character, args.output_dir, f'{frame_count:04d}.jpeg'))
            frame_count += 1
        if args.output_webcam:
            result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
                cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            cam.send(result_image)
            cam.sleep_until_next_frame()
        # toc=time.perf_counter()
        # fps =1/(toc-tic)
        # print(fps)


if __name__ == '__main__':
    main()
