from PIL import Image

import cv2
import torch
import numpy as np
import mediapipe as mp
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

from utils import get_distance

# pose vector index ------------------------------------------
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
# ------------------------------------------------------------


class Landmark:
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f'x: {self.x}\ny: {self.y}\nz: {self.z}'


def get_iris_center_point(landmarks, side=None):
    """
    calculate right or left side iris center 3d coordinate
    Args:
        landmarks:
        side:

    Returns:
        iris center 3d coordinate point (Landmark)

    Raises:
        if you didn't specify side

    """
    if side is None:
        raise 'please choose side `left` or `right`'
    elif side.lower() == 'right' or side.lower() == 'r':
        landmark_list = FACEMESH_RIGHT_IRIS
    elif side.lower() == 'left' or side.lower() == 'l':
        landmark_list = FACEMESH_LEFT_IRIS

    temp = np.zeros(3)
    for idx in list(map(lambda x: x[0], landmark_list)):
        pos_x, pos_y, pos_z = landmarks[idx].x, landmarks[idx].y, landmarks[idx].z
        temp[0] += pos_x
        temp[1] += pos_y
        temp[2] += pos_z
    temp = temp / len(landmark_list)
    return Landmark(temp[0], temp[1], temp[2])


def get_pose(facial_landmarks, debug_image=None):
    """
    extract pose vector from facial landmarks

    Args:
        facial_landmarks (landmarks): extracted facial landmarks by mediapipe
        debug_image (numpy array): frame image from video or webcam

    Returns:
        tuple: pose vector
        numpy array(optional): draw features to frame image
    """
    iris_r_center = get_iris_center_point(facial_landmarks, 'r')
    iris_l_center = get_iris_center_point(facial_landmarks, 'l')


    mouth_h = facial_landmarks[13].y - facial_landmarks[14].y
    mouth_w = facial_landmarks[78].x - (facial_landmarks[409].x + facial_landmarks[375].x) / 2
    mouth_ratio = mouth_h / mouth_w

    x_angle = np.arctan2(facial_landmarks[197].y - facial_landmarks[9].y, facial_landmarks[197].z - facial_landmarks[9].z)
    y_angle = np.arctan2(facial_landmarks[386].z - facial_landmarks[159].z, facial_landmarks[386].x - facial_landmarks[159].x)
    z_angle = np.arctan2(facial_landmarks[9].y - facial_landmarks[152].y, facial_landmarks[9].x - facial_landmarks[152].x)

    iris_rotation_l_h = get_distance(facial_landmarks[386], facial_landmarks[374])
    iris_rotation_l_w = get_distance(facial_landmarks[382], facial_landmarks[263])
    iris_rotation_r_h = get_distance(facial_landmarks[159], facial_landmarks[145])
    iris_rotation_r_w = get_distance(facial_landmarks[33], facial_landmarks[155])

    iris_rotation_l_h_temp = np.sqrt(
        (iris_l_center.x - facial_landmarks[386].x) ** 2 + (iris_l_center.y - facial_landmarks[386].y) ** 2)
    iris_rotation_l_w_temp = np.sqrt(
        (iris_l_center.x - facial_landmarks[382].x) ** 2 + (iris_l_center.y - facial_landmarks[382].y) ** 2)
    iris_rotation_r_h_temp = np.sqrt(
        (iris_r_center.x - facial_landmarks[159].x) ** 2 + (iris_r_center.y - facial_landmarks[159].y) ** 2)
    iris_rotation_r_w_temp = np.sqrt(
        (iris_r_center.x - facial_landmarks[33].x) ** 2 + (iris_r_center.y - facial_landmarks[33].y) ** 2)

    eye_x_ratio = ((iris_rotation_l_w_temp / iris_rotation_l_w + iris_rotation_r_w_temp / iris_rotation_r_w) - 1) * 3
    eye_y_ratio = ((iris_rotation_l_h_temp / iris_rotation_l_h + iris_rotation_r_h_temp / iris_rotation_r_h) - 1) * 3

    eye_l_h_temp = 1 - 2 * (facial_landmarks[145].y - facial_landmarks[159].y) / (
                facial_landmarks[155].x - facial_landmarks[33].x)
    eye_r_h_temp = 1 - 2 * (facial_landmarks[374].y - facial_landmarks[386].y) / (
                facial_landmarks[263].x - facial_landmarks[382].x)

    if debug_image is not None:
        h, w, c = debug_image.shape

        # draw iris
        cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(iris_l_center.x * w), int(iris_l_center.y * h)], (0, 128, 128), 3)
        cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(iris_r_center.x * w), int(iris_r_center.y * h)], (255, 128, 0), 3)

        # draw iris connection
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(iris_r_center.x * w), int(iris_r_center.y * h)], (0, 128, 255), 1)

        # draw eye edge2iris
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[33].x * w), int(facial_landmarks[33].y * h)], (0, 255, 128), 1)
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[145].x * w), int(facial_landmarks[145].y * h)], (0, 255, 128), 1)
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[155].x * w), int(facial_landmarks[155].y * h)], (0, 255, 128), 1)
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[159].x * w), int(facial_landmarks[159].y * h)], (0, 255, 128), 1)

        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[382].x * w), int(facial_landmarks[382].y * h)], (255, 128, 0), 1)
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[386].x * w), int(facial_landmarks[386].y * h)], (255, 128, 0), 1)
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[263].x * w), int(facial_landmarks[263].y * h)], (255, 128, 0), 1)
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[374].x * w), int(facial_landmarks[374].y * h)], (255, 128, 0), 1)

        # draw mouth
        debug_image = cv2.line(debug_image, [int((facial_landmarks[409].x + facial_landmarks[375].x) / 2 * w), int((facial_landmarks[409].y + facial_landmarks[375].y) / 2 * h)],
                 [int(facial_landmarks[78].x * w), int(facial_landmarks[78].y * h)], (255, 64, 0), 1)

        debug_image = cv2.line(debug_image, [int(facial_landmarks[13].x * w), int(facial_landmarks[13].y * h)],
                 [int(facial_landmarks[14].x * w), int(facial_landmarks[14].y * h)], (255, 64, 0), 1)

        # draw mouth
        debug_image = cv2.line(debug_image, [int(facial_landmarks[197].x * w), int(facial_landmarks[197].y * h)],
                              [int(facial_landmarks[9].x * w), int(facial_landmarks[9].y * h)], (0, 0, 255), 1)

        debug_image = cv2.flip(debug_image, 1)

        cv2.putText(debug_image, f'x angle : {x_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_image, f'y angle : {y_angle:.2f}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_image, f'z angle : {z_angle:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return (eye_l_h_temp, eye_r_h_temp, mouth_ratio, eye_y_ratio, eye_x_ratio, y_angle, z_angle, x_angle), debug_image
    else:
        return eye_l_h_temp, eye_r_h_temp, mouth_ratio, eye_y_ratio, eye_x_ratio, y_angle, z_angle, x_angle


if __name__ == '__main__':
    from utils import preprocessing_image, postprocessing_image, get_distance
    from models import TalkingAnimeLight

    model = TalkingAnimeLight().cuda()
    model = model.eval()
    model = model.half()
    img = Image.open("character/0018.png")
    img = img.resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0)

    mp_facemesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)

    facemesh = mp_facemesh.FaceMesh(refine_landmarks=True)

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

        pose, debug_image = get_pose(facial_landmarks, frame)

        if len(pose_queue) < 3:
            pose_queue.append(pose)
            pose_queue.append(pose)
            pose_queue.append(pose)
        else:
            pose_queue.pop(0)
            pose_queue.append(pose)

        np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.7, 0.2, 0.1])

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

        pose_vector[0, 0] = (x_angle - 1.5) * 1.6
        pose_vector[0, 1] = y_angle * 2.0  # temp weight
        pose_vector[0, 2] = (1.5 + z_angle) * 2  # temp weight

        output_image = model(input_image, mouth_eye_vector, pose_vector)
        cv2.imshow("frame", cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2BGRA))
        cv2.imshow("camera", debug_image)
        cv2.waitKey(1)
