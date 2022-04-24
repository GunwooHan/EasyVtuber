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
parser.add_argument('--character', type=str, default='test3')
parser.add_argument('--output_dir', type=str, default=f'dst')
parser.add_argument('--output_webcam', action='store_true')
parser.add_argument('--ifm', type=str)
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
            # print(socket_string)
            # {"nT": 0.0, "pT": 0.0, "sT": 0, "targetName": "Neutral", "animBakeFlag": 0, "count": 0, "shapeKeyValues": {"cheekSquintRight": 0.04, "mouthUpperUpRight": 0.04, "noseSneerLeft": 0.09, "mouthRight": 0.03, "mouthSmileLeft": 0.0, "eyeLookUpLeft": 0.2, "mouthLowerDownRight": 0.07, "eyeBlinkLeft": 0.01, "eyeWideLeft": 0.03, "eyeLookInLeft": 0.13, "browOuterUpRight": 0.13, "mouthShrugUpper": 0.11, "jawOpen": 0.11, "mouthPucker": 0.31, "browDownRight": 0.0, "jawLeft": 0.0, "mouthRollLower": 0.15, "eyeLookOutLeft": 0.0, "browOuterUpLeft": 0.13, "mouthStretchLeft": 0.16, "eyeLookDownRight": 0.0, "noseSneerRight": 0.09, "mouthShrugLower": 0.13, "mouthFrownLeft": 0.21, "eyeBlinkRight": 0.01, "mouthSmileRight": 0.0, "mouthDimpleRight": 0.04, "mouthUpperUpLeft": 0.04, "eyeLookUpRight": 0.2, "mouthDimpleLeft": 0.04, "mouthClose": 0.1, "cheekPuff": 0.21, "eyeSquintLeft": 0.13, "mouthFrownRight": 0.16, "eyeLookDownLeft": 0.0, "mouthFunnel": 0.34, "jawRight": 0.02, "eyeLookOutRight": 0.02, "eyeSquintRight": 0.13, "tongueOut": 0.0, "eyeLookInRight": 0.0, "mouthLowerDownLeft": 0.07, "mouthLeft": 0.0, "mouthRollUpper": 0.1, "mouthPressRight": 0.04, "mouthStretchRight": 0.18, "browInnerUp": 0.22, "mouthPressLeft": 0.03, "browDownLeft": 0.0, "eyeWideRight": 0.03, "cheekSquintLeft": 0.04, "jawForward": 0.08}, "positionValues": {"": {"tx": -0.0, "ty": 0.0, "tz": -0.0}}, "returnFPS": 0.05, "realTimeMode": true, "boneValues": {"armatures": {"armatureName": "Armature", "bones": {"headBone": {"q_w": 0.9908312276310741, "q_rx": -0.13128680667039147, "q_ry": 0.028578885406346385, "q_rz": 0.014159804176644197, "e_rx": -0.26284658535034605, "e_ry": 0.0603883921190038, "e_rz": 0.020594885173533087}, "rightEyeBone": {"q_w": 0.999019748721826, "q_rx": -0.04378889121142997, "q_ry": -0.006478228132757819, "q_rz": 0.0003274607050231633, "e_rx": -0.08761552845011535, "e_ry": -0.012915436464758038, "e_rz": 0.0012217304763960308}, "leftEyeBone": {"q_w": 0.9981535423197038, "q_rx": -0.043333803525119896, "q_ry": -0.04252150169724937, "q_rz": 0.001899821066440853, "e_rx": -0.08709192967451705, "e_ry": -0.08482300164692443, "e_rz": 0.007504915783575617}, "": {"q_w": 0.9991759923754062, "q_rx": -0.03944292952148994, "q_ry": 0.00892928988819085, "q_rz": 0.0034437412036836764, "e_rx": -0.07885397560510381, "e_ry": 0.01811651763570114, "e_rz": 0.006178465552059927}}}}}
            blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(blender_data)
            cur_time = time.perf_counter()
            fps = 1 / (cur_time - self.perf_time)
            self.perf_time = cur_time
            # print("F:",fps)
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


class ClientProcess1(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = args.ifm.split(':')[0]
        self.port = int(args.ifm.split(':')[1])
        self.perf_time = 0

    def run(self):

        udpClntSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        data = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719"

        data = data.encode('utf-8')

        udpClntSock.sendto(data, (self.address, self.port))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(("", self.port))
        self.socket.settimeout(0.1)
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK or err == 'timed out':
                    continue
                else:
                    raise e
            socket_string = socket_bytes.decode("utf-8")
            # print(socket_string)

            # blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(socket_string)
            # cur_time = time.perf_counter()
            # fps = 1 / (cur_time - self.perf_time)
            # self.perf_time = cur_time
            # print(fps)
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        for item in blender_data.split('|'):
            if (item.find('#') != -1):
                k, arr = item.split('#')
                arr = [float(n) for n in arr.split(',')]
                data[k.replace("_L", "Left").replace("_R", "Right")] = arr
            elif (item.find('-') != -1):
                k, v = item.split("-")
                data[k.replace("_L", "Left").replace("_R", "Right")] = float(v) / 100

        # shape_key_values = blender_data["shapeKeyValues"]
        # for blendshape_name in BLENDSHAPE_NAMES:
        #     data[blendshape_name] = shape_key_values[blendshape_name]
        #
        # head_bone = blender_data["boneValues"]["armatures"]["bones"]["headBone"]
        toRad = 57.3
        data[HEAD_BONE_X] = data["=head"][0] / toRad
        data[HEAD_BONE_Y] = data["=head"][1] / toRad
        data[HEAD_BONE_Z] = data["=head"][2] / toRad
        # data[HEAD_BONE_QUAT] = [
        #     head_bone["q_rx"],
        #     head_bone["q_ry"],
        #     head_bone["q_rz"],
        #     head_bone["q_w"],
        # ]
        #
        # right_eye_bone = blender_data["boneValues"]["armatures"]["bones"]["rightEyeBone"]
        data[RIGHT_EYE_BONE_X] = data["rightEye"][0] / toRad
        data[RIGHT_EYE_BONE_Y] = data["rightEye"][1] / toRad
        data[RIGHT_EYE_BONE_Z] = data["rightEye"][2] / toRad
        # data[RIGHT_EYE_BONE_QUAT] = [
        #     right_eye_bone["q_rx"],
        #     right_eye_bone["q_ry"],
        #     right_eye_bone["q_rz"],
        #     right_eye_bone["q_w"],
        # ]
        #
        # left_eye_bone = blender_data["boneValues"]["armatures"]["bones"]["leftEyeBone"]
        data[LEFT_EYE_BONE_X] = data["leftEye"][0] / toRad
        data[LEFT_EYE_BONE_Y] = data["leftEye"][1] / toRad
        data[LEFT_EYE_BONE_Z] = data["leftEye"][2] / toRad
        # data[LEFT_EYE_BONE_QUAT] = [
        #     left_eye_bone["q_rx"],
        #     left_eye_bone["q_ry"],
        #     left_eye_bone["q_rz"],
        #     left_eye_bone["q_w"],
        # ]

        return data


@torch.no_grad()
def main():
    model = TalkingAnimeLight().to(device)
    model = model.eval()
    model = model
    img = Image.open(f"character/{args.character}.png")
    img = img.resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0)

    client_process = ClientProcess1()
    client_process.start()

    # client_process1 = ClientProcess1()
    # client_process1.start()

    # if args.input == 'cam':
    #     cap = cv2.VideoCapture(1)
    #     ret, frame = cap.read()
    #     if ret is None:
    #         raise Exception("Can't find Camera")
    # else:
    #     cap = cv2.VideoCapture(args.input)
    #     frame_count = 0
    #     os.makedirs(os.path.join('dst', args.character, args.output_dir), exist_ok=True)

    # facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    if args.output_webcam:
        cam = pyvirtualcam.Camera(width=256, height=256, fps=30, backend='unitycapture',
                                  fmt=pyvirtualcam.PixelFormat.RGBA)
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
        pose_vector[0, 0] = 0
        pose_vector[0, 1] = 0
        pose_vector[0, 2] = 0

        # debugValue=time.perf_counter()%1
        #
        # mouth_eye_vector[0, 22] = debugValue
        # mouth_eye_vector[0, 23] = debugValue

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
            # result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            # result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
            #     cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            result_image = postprocessing_image(output_image.cpu())
            cam.send(result_image)
            cam.sleep_until_next_frame()
        toc = time.perf_counter()
        fps = 1 / (toc - tic)
        # print("R:",fps)


if __name__ == '__main__':
    main()
