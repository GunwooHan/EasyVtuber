# from pyanime4k import ac
# import pyanime4k
#
# parameters = ac.Parameters()
# # enable HDN for ACNet
# parameters.HDN = True
#
# a = ac.AC(
#     managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=0, dID=0)]),
#     type=ac.ProcessorType.OpenCL_ACNet
# )
#
# import cv2
#
# img = cv2.imread("character/test41.png")
#
# a.load_image_from_numpy(img,input_type=ac.AC_INPUT_BGR)
#
# # start processing
# a.process()
#
# # preview upscaled image
# img=a.save_image_to_numpy()
#
# print(img)
import cv2
import numpy as np

rgb=[
    [[1,2,3],[2,2,3],[3,2,3],[4,2,3]],
    [[1,2,3],[2,2,3],[3,2,3],[4,2,3]],
    [[1,2,3],[2,2,3],[3,2,3],[4,2,3]],
    [[1,2,3],[2,2,3],[3,2,3],[4,2,3]]
]
a=[
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
]

print(cv2.merge((np.array(rgb),np.array(a))))