import cv2
import numpy as np
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

from facial_points import MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT1, MOUTH_LEFT2, MOUTH_RIGHT
from facial_points import IRIS_L_TOP, IRIS_L_BOTTOM, IRIS_L_LEFT, IRIS_L_RIGHT
from facial_points import IRIS_R_TOP, IRIS_R_BOTTOM, IRIS_R_LEFT, IRIS_R_RIGHT
from utils import get_distance


# pose vector index ------------------------------------------
# wink                          0, 1
# happy wink                    2, 3
# surprised                     4, 5
# relaxed                       6, 7
# unimpressed                   8, 9
# iris Shrinkage L              12
# iris Shrinkage R              MOUTH_TOP
# mouth aaa                     MOUTH_BOTTOM
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

# def set_pose(pose, pose_vector)


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

    mouth_h = facial_landmarks[MOUTH_TOP].y - facial_landmarks[MOUTH_BOTTOM].y
    mouth_w = facial_landmarks[MOUTH_RIGHT].x - (facial_landmarks[MOUTH_LEFT1].x + facial_landmarks[MOUTH_LEFT2].x) / 2
    mouth_ratio = mouth_h / mouth_w

    x_angle = np.arctan2(facial_landmarks[197].y - facial_landmarks[9].y, facial_landmarks[197].z - facial_landmarks[9].z)
    y_angle = np.arctan2(facial_landmarks[IRIS_L_TOP].z - facial_landmarks[IRIS_R_TOP].z, facial_landmarks[IRIS_L_TOP].x - facial_landmarks[IRIS_R_TOP].x)
    z_angle = np.arctan2(facial_landmarks[9].y - facial_landmarks[152].y, facial_landmarks[9].x - facial_landmarks[152].x)

    iris_rotation_l_h = get_distance(facial_landmarks[IRIS_L_TOP], facial_landmarks[IRIS_L_BOTTOM])
    iris_rotation_l_w = get_distance(facial_landmarks[IRIS_L_RIGHT], facial_landmarks[IRIS_L_LEFT])
    iris_rotation_r_h = get_distance(facial_landmarks[IRIS_R_TOP], facial_landmarks[IRIS_R_BOTTOM])
    iris_rotation_r_w = get_distance(facial_landmarks[IRIS_R_RIGHT], facial_landmarks[IRIS_R_LEFT])

    iris_rotation_l_h_temp = np.sqrt(
        (iris_l_center.x - facial_landmarks[IRIS_L_TOP].x) ** 2 + (iris_l_center.y - facial_landmarks[IRIS_L_TOP].y) ** 2)
    iris_rotation_l_w_temp = np.sqrt(
        (iris_l_center.x - facial_landmarks[IRIS_L_RIGHT].x) ** 2 + (iris_l_center.y - facial_landmarks[IRIS_L_RIGHT].y) ** 2)
    iris_rotation_r_h_temp = np.sqrt(
        (iris_r_center.x - facial_landmarks[IRIS_R_TOP].x) ** 2 + (iris_r_center.y - facial_landmarks[IRIS_R_TOP].y) ** 2)
    iris_rotation_r_w_temp = np.sqrt(
        (iris_r_center.x - facial_landmarks[IRIS_R_RIGHT].x) ** 2 + (iris_r_center.y - facial_landmarks[IRIS_R_RIGHT].y) ** 2)

    eye_x_ratio = ((iris_rotation_l_w_temp / iris_rotation_l_w + iris_rotation_r_w_temp / iris_rotation_r_w) - 1) * 3
    eye_y_ratio = ((iris_rotation_l_h_temp / iris_rotation_l_h + iris_rotation_r_h_temp / iris_rotation_r_h) - 1) * 3

    eye_l_h_temp = 1 - 2 * (facial_landmarks[IRIS_R_BOTTOM].y - facial_landmarks[IRIS_R_TOP].y) / (
                facial_landmarks[IRIS_R_LEFT].x - facial_landmarks[IRIS_R_RIGHT].x)
    eye_r_h_temp = 1 - 2 * (facial_landmarks[IRIS_L_BOTTOM].y - facial_landmarks[IRIS_L_TOP].y) / (
                facial_landmarks[IRIS_L_LEFT].x - facial_landmarks[IRIS_L_RIGHT].x)

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
                 [int(facial_landmarks[IRIS_R_RIGHT].x * w), int(facial_landmarks[IRIS_R_RIGHT].y * h)], (0, 255, 128), 1)
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[IRIS_R_BOTTOM].x * w), int(facial_landmarks[IRIS_R_BOTTOM].y * h)], (0, 255, 128), 1)
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[IRIS_R_LEFT].x * w), int(facial_landmarks[IRIS_R_LEFT].y * h)], (0, 255, 128), 1)
        debug_image = cv2.line(debug_image, [int(iris_r_center.x * w), int(iris_r_center.y * h)],
                 [int(facial_landmarks[IRIS_R_TOP].x * w), int(facial_landmarks[IRIS_R_TOP].y * h)], (0, 255, 128), 1)

        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[IRIS_L_RIGHT].x * w), int(facial_landmarks[IRIS_L_RIGHT].y * h)], (255, 128, 0), 1)
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[IRIS_L_TOP].x * w), int(facial_landmarks[IRIS_L_TOP].y * h)], (255, 128, 0), 1)
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[IRIS_L_LEFT].x * w), int(facial_landmarks[IRIS_L_LEFT].y * h)], (255, 128, 0), 1)
        debug_image = cv2.line(debug_image, [int(iris_l_center.x * w), int(iris_l_center.y * h)],
                 [int(facial_landmarks[IRIS_L_BOTTOM].x * w), int(facial_landmarks[IRIS_L_BOTTOM].y * h)], (255, 128, 0), 1)

        # draw mouth
        debug_image = cv2.line(debug_image, [int((facial_landmarks[MOUTH_LEFT1].x + facial_landmarks[MOUTH_LEFT2].x) / 2 * w), int((facial_landmarks[MOUTH_LEFT1].y + facial_landmarks[MOUTH_LEFT2].y) / 2 * h)],
                 [int(facial_landmarks[MOUTH_RIGHT].x * w), int(facial_landmarks[MOUTH_RIGHT].y * h)], (255, 64, 0), 1)

        debug_image = cv2.line(debug_image, [int(facial_landmarks[MOUTH_TOP].x * w), int(facial_landmarks[MOUTH_TOP].y * h)],
                 [int(facial_landmarks[MOUTH_BOTTOM].x * w), int(facial_landmarks[MOUTH_BOTTOM].y * h)], (255, 64, 0), 1)

        # draw mouth
        debug_image = cv2.line(debug_image, [int(facial_landmarks[197].x * w), int(facial_landmarks[197].y * h)],
                              [int(facial_landmarks[9].x * w), int(facial_landmarks[9].y * h)], (0, 0, 255), 1)

        debug_image = cv2.flip(debug_image, 1)

        cv2.putText(debug_image, f'x angle : {x_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_image, f'y angle : {y_angle:.2f}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_image, f'z angle : {z_angle:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return (eye_l_h_temp, eye_r_h_temp, mouth_ratio, eye_y_ratio, eye_x_ratio, x_angle , y_angle, z_angle), debug_image
    else:
        return eye_l_h_temp, eye_r_h_temp, mouth_ratio, eye_y_ratio, eye_x_ratio,x_angle, y_angle, z_angle