# from pyanime4k import ac
# import pyanime4k
#
# parameters = ac.Parameters()
# # enable HDN for ACNet
# parameters.HDN = True
#
# a = ac.AC(
#     type=ac.ProcessorType.CPU_ACNet,
# )
# a.set_arguments(parameters)
import struct

import cv2
#
# img = cv2.imread("character/test41.png")
#
# a.load_image_from_numpy(img,input_type=ac.AC_INPUT_BGR)
#
# a.get_processor_info()
#
# # start processing
# a.process()
#
# a.show_image()

# import numpy as np
# import zlib
# import bz2
# import lzma
# import time
#
# postprocessed_image=np.load('out.npy')
# tic=time.perf_counter()
# res=None
# for i in range(100):
#     res=zlib.compress(postprocessed_image)
# print("zlib", len(res),(time.perf_counter() - tic) * 1000)
# tic = time.perf_counter()
# for i in range(100):
#     zlib.decompress(res)
# print("zlib", len(res),(time.perf_counter() - tic) * 1000)
# tic = time.perf_counter()
# for i in range(100):
#     res=bz2.compress(postprocessed_image)
# print("bz2",len(res), (time.perf_counter() - tic) * 1000)
# tic = time.perf_counter()
# for i in range(100):
#     bz2.decompress(res)
# print("bz2", len(res),(time.perf_counter() - tic) * 1000)
# tic = time.perf_counter()
# for i in range(100):
#     res=lzma.compress(postprocessed_image)
# print("lzma",len(res), (time.perf_counter() - tic) * 1000)
# tic = time.perf_counter()
# for i in range(100):
#     lzma.decompress(res)
# print("lzma", len(res),(time.perf_counter() - tic) * 1000)
# tic = time.perf_counter()
#
#
# output_frame = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGRA)
# cv2.imshow("frame", output_frame)
# cv2.waitKey(100000)
# preview upscaled image
# img=a.save_image_to_numpy()
#
# print(img)
# import cv2
# import numpy as np
#
# rgb=[
#     [[1,2,3],[2,2,3],[3,2,3],[4,2,3]],
#     [[1,2,3],[2,2,3],[3,2,3],[4,2,3]],
#     [[1,2,3],[2,2,3],[3,2,3],[4,2,3]],
#     [[1,2,3],[2,2,3],[3,2,3],[4,2,3]]
# ]
# a=[
#     [1,2,3,4],
#     [1,2,3,4],
#     [1,2,3,4],
#     [1,2,3,4],
# ]
#
# print(cv2.merge((np.array(rgb),np.array(a))))


from multiprocessing import Value, Process, Queue
import queue
import socket
import errno
import time
import tha2.poser.modes.mode_20_wx
from tha2.mocap.ifacialmocap_constants import *
import numpy as np

ifm='127.0.0.1:11573'



class ClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = ifm.split(':')[0]
        self.port = int(ifm.split(':')[1])
        self.perf_time = 0
        # self.ifm_converter = tha2.poser.modes.mode_20_wx.create_ifacialmocap_pose_converter()
        self.a_min=None
        self.a_max=None

    def run(self):
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

            # socket_string = socket_bytes.decode("utf-8")
            osf_raw=(struct.unpack('=di2f2fB1f4f3f3f68f136f210f14f',socket_bytes))
            # print(osf_raw[432:])
            data={}
            OpenSeeDataIndex=[
                'time',
                'id',
                'cameraResolutionW',
                'cameraResolutionH',
                'rightEyeOpen',
                'leftEyeOpen',
                'got3DPoints',
                'fit3DError',
                'rawQuaternionX',
                'rawQuaternionY',
                'rawQuaternionZ',
                'rawQuaternionW',
                'rawEulerX',
                'rawEulerY',
                'rawEulerZ',
                'translationY',
                'translationX',
                'translationZ',
            ]
            for i in range(len(OpenSeeDataIndex)):
                data[OpenSeeDataIndex[i]]=osf_raw[i]
            data['translationY']*=-1
            data['translationZ']*=-1
            data['rotationY']=data['rawEulerY']
            data['rotationX']=(-data['rawEulerX']+180)%360
            data['rotationZ']=(data['rawEulerZ']-90)
            OpenSeeFeatureIndex=[
                'EyeLeft',
                'EyeRight',
                'EyebrowSteepnessLeft',
                'EyebrowUpDownLeft',
                'EyebrowQuirkLeft',
                'EyebrowSteepnessRight',
                'EyebrowUpDownRight',
                'EyebrowQuirkRight',
                'MouthCornerUpDownLeft',
                'MouthCornerInOutLeft',
                'MouthCornerUpDownRight',
                'MouthCornerInOutRight',
                'MouthOpen',
                'MouthWide'
            ]

            for i in range(68):
                data['confidence'+str(i)]=osf_raw[i+18]
            for i in range(68):
                data['pointsX'+str(i)]=osf_raw[i*2+18+68]
                data['pointsY'+str(i)]=osf_raw[i*2+18+68+1]
            for i in range(70):
                data['points3DX'+str(i)]=osf_raw[i*3+18+68+68*2]
                data['points3DY'+str(i)]=osf_raw[i*3+18+68+68*2+1]
                data['points3DZ'+str(i)]=osf_raw[i*3+18+68+68*2+2]

            for i in range(len(OpenSeeFeatureIndex)):
                data[OpenSeeFeatureIndex[i]]=osf_raw[i+432]
            # print(data['rotationX'],data['rotationY'],data['rotationZ'])

            a=np.array([
                data['points3DX66']-data['points3DX68']+data['points3DX67']-data['points3DX69'],
                data['points3DY66']-data['points3DY68']+data['points3DY67']-data['points3DY69'],
                data['points3DZ66']-data['points3DZ68']+data['points3DZ67']-data['points3DZ69']
            ])
            a=(a/np.linalg.norm(a))
            data['eyeRotationX']=a[0]
            data['eyeRotationY']=a[1]
            # blender_data = json.loads(socket_string)
            # data = self.convert_from_blender_data(socket_string)
            # data=self.ifm_converter.convert(data)
            # print(data)
            # if(self.a_max is None):
            #     self.a_max=[x for x in data]
            # else:
            #     self.a_max=[max(data[i],self.a_max[i]) for i in range(len(data))]
            # if(self.a_min is None):
            #     self.a_min=[x for x in data]
            # else:
            #     self.a_min=[min(data[i],self.a_min[i]) for i in range(len(data))]
            # print([[self.a_min[i],self.a_max[i]] for i in range(len(data))])
            cur_time = time.perf_counter()
            fps = 1 / (cur_time - self.perf_time)
            self.perf_time = cur_time
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
            if item.find('#') != -1:
                k, arr = item.split('#')
                arr = [float(n) for n in arr.split(',')]
                data[k.replace("_L", "Left").replace("_R", "Right")] = arr
            elif item.find('-') != -1:
                k, v = item.split("-")
                data[k.replace("_L", "Left").replace("_R", "Right")] = float(v) / 100

        to_rad = 57.3
        data[HEAD_BONE_X] = data["=head"][0] / to_rad
        data[HEAD_BONE_Y] = data["=head"][1] / to_rad
        data[HEAD_BONE_Z] = data["=head"][2] / to_rad
        data[HEAD_BONE_QUAT] = [data["=head"][3], data["=head"][4], data["=head"][5], 1]
        # print(data[HEAD_BONE_QUAT][2],min(data[EYE_BLINK_LEFT],data[EYE_BLINK_RIGHT]))
        data[RIGHT_EYE_BONE_X] = data["rightEye"][0] / to_rad
        data[RIGHT_EYE_BONE_Y] = data["rightEye"][1] / to_rad
        data[RIGHT_EYE_BONE_Z] = data["rightEye"][2] / to_rad
        data[LEFT_EYE_BONE_X] = data["leftEye"][0] / to_rad
        data[LEFT_EYE_BONE_Y] = data["leftEye"][1] / to_rad
        data[LEFT_EYE_BONE_Z] = data["leftEye"][2] / to_rad

        return data


if __name__ == '__main__':
    client_process = ClientProcess()
    client_process.daemon = True
    client_process.start()

    while True:
        pass