# USAGE
# python intersection_over_union.py

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import scipy.io
import yaml
import sys

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# define the list of example detections
# examples = [
#	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
#	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
#	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
#	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
#	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]
# read .mat is impossible
# detection= scipy.io.loadmat('/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_lab/ground_truth_image_plane.mat')
# file_GT = open('/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_corridor/ground_truth_image_plane.yaml')
file_GT = open('/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_lab/ground_truth_image_plane.yaml')
file_video = open('/home/carlos/Escritorio/dataset/IOU/intersection-over-union/result.yaml')

file_dict_GT = yaml.load(file_GT, Loader=yaml.FullLoader)
file_dict_video = yaml.load(file_video, Loader=yaml.FullLoader)
# frames_dict_GT=file_dict_GT['20141008_141323_00']
frames_dict_GT = file_dict_GT['20140804_160621_00']
frames_dict_video = file_dict_video['lab_video']

initial = 0  # next(iter(frames_dict_GT))

# extraccion del video
camera_width = 512
camera_height = 424
vidfps = 25
number_frame = 0
PATH = '/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_lab/20140804_160621_00/out_epfl_rgbd.mp4'
# PATH='/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_corridor/20141008_141323_00/video.mp4'
cam = cv2.VideoCapture(PATH)
if cam.isOpened() != True:
    print("USB Camera Open Error!!!")
    sys.exit(0)

cam.set(cv2.CAP_PROP_FPS, vidfps)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
framenumber = 0

num_of_frame = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow("USB_camera", cv2.WINDOW_AUTOSIZE)
dict_IOU = []
list_IOU = []
# read .yaml from dictionary

estado = []
initial = (next(iter(frames_dict_GT)))
# loop over the example detections
while (initial < (num_of_frame - (next(iter(frames_dict_GT))))):

    s, image = cam.read()
    flag_image = image

    if initial >= (next(iter(frames_dict_GT))):

        sel_GT = frames_dict_GT[initial]
        sel_video = frames_dict_video[initial + 13]
        k = 0

        list_IOU_0 = [0, 0, 0, 0, 0, 0, 0]
        list_IOU_1 = [0, 0, 0, 0, 0, 0, 0]
        list_IOU_2 = [0, 0, 0, 0, 0, 0, 0]
        list_IOU_3 = [0, 0, 0, 0, 0, 0, 0]
        estado = [0, 0, 0, 0]

        for l in list(sel_GT.keys()):
            # k=next(iter(sel_GT))+l

            # image=flag_image
            box_gT_dict = sel_GT[l]
            box_gT_list = list(box_gT_dict)
            box_gt_func = [box_gT_list[0], box_gT_list[1], box_gT_list[0] + box_gT_list[2],
                           box_gT_list[1] + box_gT_list[3]]

            for i in list(sel_video.keys()):

                image = flag_image
                image = cv2.putText(image, str(initial), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.circle(image, (0, 0), 2, (0, 0, 255), 2)
                cv2.circle(image, (256, 212), 2, (0, 0, 255), 2)
                cv2.circle(image, (box_gT_list[0], box_gT_list[1]), 2, (0, 0, 255), 2)
                image = cv2.rectangle(image, (box_gT_list[0], box_gT_list[1]),
                                      (box_gT_list[0] + box_gT_list[2], box_gT_list[1] + box_gT_list[3]),
                                      (0, 250, (l * 50)), 2)
                box_video_dict = sel_video[i]
                box_video_list = list(box_video_dict)

                # color BGR

                image = cv2.rectangle(image, (box_video_list[0], box_video_list[1]),
                                      (box_video_list[2], box_video_list[3]), (255, 0, (i * 50)), 2)

                if l == 0:
                    list_IOU_0[i] = bb_intersection_over_union(box_gt_func, box_video_list)
                    estado[l] = 1
                elif l == 1:
                    list_IOU_1[i] = bb_intersection_over_union(box_gt_func, box_video_list)
                    estado[l] = 1
                elif l == 2:
                    list_IOU_2[i] = bb_intersection_over_union(box_gt_func, box_video_list)
                    estado[l] = 1
                elif l == 3:
                    list_IOU_3[i] = bb_intersection_over_union(box_gt_func, box_video_list)
                    estado[l] = 1

        """""cv2.imshow("USB_camera", image)
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0) """""

        if initial == 34:
            dict_IOU = {initial: {0: list_IOU_0, 1: list_IOU_1, 2: list_IOU_2, 3: list_IOU_3, "estado": estado}}
        elif initial > 34:
            dict_IOU.update({initial: {0: list_IOU_0, 1: list_IOU_1, 2: list_IOU_2, 3: list_IOU_3, "estado": estado}})

        # funcion IOU

    initial += 1
    # BGR
# Procesado del diccionarion
print("Procesando de estadisticas")
FP_list = []
TP_list = []
FN_list = []
TN_list = []

FP = 0
TP = 0
FN = 0
TN = 0
dict_resutl = []

# comprobar la IOU
n = (next(iter(frames_dict_GT)))
# first input
FP_dic = {n: {"FP0": 0, "FP1": 0, "FP2": 0, "FP3": 0}}
TP_dic = {n: {"TP0": 0, "TP1": 0, "TP2": 0, "TP3": 0}}
FN_dic = {n: {"FN0": 0, "FN1": 0, "FN2": 0, "FN3": 0}}
TN_dic = {n: {"TN0": 0, "TN1": 0, "TN2": 0, "TN3": 0}}
while n < (num_of_frame - (next(iter(frames_dict_GT)))):

    if n >= (next(iter(frames_dict_GT))):

        #print("frame numero: ", n)
        frame_information = dict_IOU.get(n)
        personas_por_GT = 0
        FP_list = [0,0,0,0]
        TP_list = [0,0,0,0]
        FN_list = [0,0,0,0]
        TN_list = [0,0,0,0]
        for i in range(len(frame_information['estado'])):


            if frame_information['estado'][i] == 1:
                personas_por_GT += 1
                identificador_persona_GT = i
                if len(set(frame_information[i])) == 1:
                    FN += 1
                else:
                    number_positive = 0
                    down_umbral = 0
                    for l in range(len(frame_information[i])):
                        if frame_information[i][l] >= 0.18:
                            number_positive += 1
                        elif 0 < frame_information[i][l] < 0.18:
                            down_umbral += 1

                    if number_positive >= 1:
                        FP += (number_positive - 1)
                        TP += 1
                    if down_umbral >= 1:
                        # FN+=1
                        FN = FN
            else:

                if len(set(frame_information[i])) == 1:

                    TN += 1
                else:
                    FP += 1
            FP_list[i]=FP
            TP_list[i]=TP
            FN_list[i]=FN
            TN_list[i]=TN

    FP_dic.update({n: {"FP0": FP_list[0], "FP1": FP_list[1], "FP2": FP_list[2], "FP3": FP_list[3]}})
    TP_dic.update({n: {"TP0": TP_list[0], "TP1": TP_list[1], "TP2": TP_list[2], "TP3": TP_list[3]}})
    FN_dic.update({n: {"FN0": FN_list[0], "FN1": FN_list[1], "FN2": FN_list[2], "FN3": FN_list[3]}})
    TN_dic.update({n: {"TN0": TN_list[0], "TN1": TN_list[1], "TN2": TN_list[2], "TN3": TN_list[3]}})
    n+=1


print("hola")