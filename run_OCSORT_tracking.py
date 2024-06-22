
#############
#####
##
# I had to run the YOLO detector first: run_YOLOv8_inference.py
##
#####
#############


from ultralytics import YOLO
from pathlib import Path
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
from datetime import datetime
import glob

# Hay que importar el tracker de OC-SORT
# para ello hay que hacer: git clone https://github.com/noahcao/OC_SORT.git
# https://github.com/noahcao/OC_SORT
from trackers.OC_SORT.trackers.ocsort_tracker.ocsort import *


# Get video dimensions for OC-SORT 
VIDEO_PATH = "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

GENERATE_MOT_FILE = True

#NUM_MAX_ANTS = 10000

def clic(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Posición del pixel: ({x}, {y})')


det_thresh = 0.5
max_age = 30
min_hits = 3
iou_threshold = 0.3
delta_t = 3
asso_func = "giou"
inertia = 0.2
use_byte = False

ocsort_tracker = OCSort(det_thresh=det_thresh, max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, delta_t=delta_t, asso_func=asso_func, inertia=inertia, use_byte=use_byte)
MOT_FILE = "/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/resultsEvalPerMemoria_iouSplit-0.60/ant_subset_1-024_YOLOv8n_Marc_MOT.txt"

directory_out_MOT_OCSORT = MOT_FILE.replace("MOT.txt", "MOT_OCSORT.txt")


#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi

# Si volem visualitzar les deteccions en el video (no implementat) podem fer servir aquesta llista de colors
random_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255), (0,0,0), (100,100,100), (200,200,200)]       

print("Start time: " + str(datetime.now()), flush=True)

MOT = np.loadtxt(MOT_FILE, delimiter=",")
frames = np.unique(MOT[:,0]).astype(int)
min_frame = np.min(frames)
max_frame = np.max(frames)
num_frames = len(frames)


for frame in range(min_frame, max_frame + 1):
    list_to_print = []
    print("Frame: " + str(frame), flush=True)
    mask = MOT[:,0] == frame
    rows = MOT[mask]
    bboxes = rows[:,2:7]

    for bbox in bboxes:
        print(bbox, flush=True)
        minx = bbox[0]
        miny = bbox[1]
        maxx = bbox[2] + minx
        maxy = bbox[3] + miny
        score = bbox[4]

        list_to_print.append([maxx, maxy, minx, miny, score])

    ############################################################################################################
    #   OC-SORT   -------------------------------------------------------------------------------------------  #
    ############################################################################################################
    list_to_OC_SORT = []
    for data in list_to_print:
        maxx, maxy, minx, miny, score = data
        list_to_OC_SORT.append([minx, miny, maxx, maxy, score])

    print(MOT_FILE, (width, height), (width, height), flush=True)
    ocsort_tracker.update(np.array(list_to_OC_SORT), img_info=(width, height), img_size=(width, height))
    #ocsort_tracker.update(np.array(list_to_OC_SORT))

    new_list_to_print = []
    list_of_ids = []
    all_MOTs_ID = [] 
    #paths = []
    #for _ in range(NUM_MAX_ANTS):
    #    paths.append([]) 

    for detection in ocsort_tracker.trackers:
        if detection.age != 0:
            list_of_ids.append(detection.id)
            #area = detection.bbox.area
            maxx = detection.last_observation[2]
            maxy = detection.last_observation[3]
            minx = detection.last_observation[0]
            miny = detection.last_observation[1]
            #shift_x = detection.bbox.shift_x
            #shift_y = detection.bbox.shift_y
            score = detection.last_observation[4]
            class_id = detection.id
            #class_name = detection.category.name
            new_list_to_print.append([None, maxx, maxy, minx, miny, None, None, score, class_id, None])

            # 0: frame_cnt, 1: class_id, 2: minx, 3: miny, 4: width, 5: height, 6: score, 7: x, 8: y, 9: z
            all_MOTs_ID.append((frame, class_id, minx, miny, maxx-minx, maxy-miny, score, -1, -1, -1))

    ## rellenem els paths amb -1 si no hi ha data
    #if len(list_of_ids) > 0:
    #    for i in range(NUM_MAX_ANTS):
    #        if i not in list_of_ids:
    #            paths[i].append((-1,-1,-1,-1,-1,-1,-1,-1))


    ### GENERAR MOT FILE AMB TOTES LES DETECCIONS A CADA FRAME + ID del tracker OC-SORT
    if GENERATE_MOT_FILE:     

        with open(directory_out_MOT_OCSORT, 'a') as file: 
            for i in all_MOTs_ID:
                string = ""

                for j in i:
                    string += str(j) + ","

                string = string[:-1]
                string += "\n"
                file.write(string)
        file.close()





print("End time: " + str(datetime.now()))