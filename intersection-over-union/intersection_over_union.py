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
#examples = [
#	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
#	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
#	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
#	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
#	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]
#read .mat is impossible
#detection= scipy.io.loadmat('/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_lab/ground_truth_image_plane.mat')

file_GT = open('/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_lab/ground_truth_image_plane.yaml')
file_video= open('/home/carlos/Escritorio/dataset/IOU/intersection-over-union/result.yaml')

file_dict_GT=yaml.load(file_GT,Loader=yaml.FullLoader)
file_dict_video= yaml.load(file_video,Loader=yaml.FullLoader)

frames_dict_GT=file_dict_GT['20140804_160621_00']
frames_dict_video=file_dict_video['lab_video']

initial=0#next(iter(frames_dict_GT))

#extraccion del video
camera_width = 640
camera_height = 480
vidfps = 25
number_frame=0
PATH='/home/carlos/Escritorio/dataset/pasillo/epfl_rgbd_pedestrians/epfl_lab/20140804_160621_00/out_epfl_rgbd.mp4'
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


#read .yaml from dictionary


# loop over the example detections
while(1):

	s, image = cam.read()

	if initial >= (next(iter(frames_dict_GT))):

		sel_GT = frames_dict_GT[initial]
		sel_video = frames_dict_video[initial+13]
		k=0
		for l in list(sel_GT.keys()):
			#k=next(iter(sel_GT))+l

			box_gT_dict = sel_GT[l]
			box_gT_list = list(box_gT_dict)
			image = cv2.rectangle(image, (box_gT_list[0], box_gT_list[1]), (box_gT_list[0] + box_gT_list[2], box_gT_list[1] + box_gT_list[3]), (255,0, (l*50)), 2)


		for i in list(sel_video.keys()):
			box_video_dict=sel_video[i]
			box_video_list=list(box_video_dict)

			#color BGR

			image = cv2.rectangle(image,(box_video_list[0],box_video_list[1]),(box_video_list[2],box_video_list[3]),(255,0, (i*50)),2)

			#funcion IOU

	initial+=1
	#BGR
	cv2.imshow("USB_camera", image)
	key = cv2.waitKey(0)
	while key not in [ord('q'), ord('k')]:
		key = cv2.waitKey(0)

	#comprobar la IOU


