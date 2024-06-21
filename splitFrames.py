import cv2
import numpy as np
import shapely.geometry as sg
from numpy import random
import matplotlib.pyplot as plt

# variables temporals, se poden eliminar
list_a = []

import numpy as np

# SPLIT FILL THE NON 640X640 FRAMES WITH BLACK BACKGROUND
def splitFrame(frame, ROI, overlap=0.2):
    # Obtindre ample i alçada del frame (normalment sera 4000x2992)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    ROI_overlap = int(ROI * overlap)

    roi_dict = {}
    
    cont = 0
    for x in range(0, frame_width, ROI - ROI_overlap):
        for y in range(0, frame_height, ROI - ROI_overlap):
            roi = frame[y:y + ROI, x:x + ROI]
            
            # If the extracted ROI is smaller than the desired size, pad it with black pixels
            if roi.shape[0] < ROI or roi.shape[1] < ROI:
                padded_roi = np.zeros((ROI, ROI, frame.shape[2]), dtype=frame.dtype)
                padded_roi[:roi.shape[0], :roi.shape[1], :] = roi
                roi = padded_roi
            #print("\nROI: ", roi.shape)
            #print((cont, x, y))
            #cv2.imshow("ROI", roi)
            #cv2.waitKey(0)
            roi_dict[(cont, x, y)] = roi
            cont += 1

    return roi_dict


def joinROIs(originalFrame, roi_dict, frameMOT, frame_width, frame_height):
    originalFrame_ = originalFrame.copy()
    # Create a blank frame
    frame = np.zeros((frame_height, frame_width, 3), np.uint8)
    max_width = 0
    max_height = 0

    ##############################################################################################
    ###  Busco duplicats en frameMOT (me quedare les deteccions que tinguin un IoU molt gran)  ###
    ##############################################################################################
    to_remove = []
    iteracions = 0
    for start in range(len(frameMOT)):
        sub_list = frameMOT[start:]
        iteracions += 1
        print("Iteracio: ", iteracions, " de ", len(frameMOT))

        for bbox_1 in sub_list:
            x1, y1, w1, h1, score1 = bbox_1[2:]
            bbox1 = sg.box(x1, y1, x1 + w1, y1 + h1)

            x1 = int(x1)
            y1 = int(y1)
            w1 = int(w1)
            h1 = int(h1)
            score1 = np.round(float(score1),2)

            if w1 > max_width:
                max_width = w1
            if h1 > max_height:
                max_height = h1

            for bbox_2 in sub_list:
                if bbox_1 == bbox_2:
                    continue
                else:                
                    x2, y2, w2, h2, score2 = bbox_2[2:]                
                    bbox2 = sg.box(x2, y2, x2 + w2, y2 + h2)              

                    x2 = int(x2)
                    y2 = int(y2)
                    w2 = int(w2)
                    h2 = int(h2)
                    score2 = np.round(float(score2),2)

                    if bbox1.distance(bbox2) < 500:

                        intersection = bbox1.intersection(bbox2)
                        union = bbox1.union(bbox2)
                        IoU = intersection.area / union.area

                        # Si alguna deteccio la fa doble, s'ha de baixar aquest valor
                        if IoU > 0.75:
                            if bbox1.area > bbox2.area:
                                if bbox2 not in to_remove:
                                    to_remove.append(bbox_2)
                            else:
                                if bbox1 not in to_remove:
                                    to_remove.append(bbox_1)
                        else:
                            pass
                            #if IoU > 0.78:
                            #    x1 = int(x1)
                            #    y1 = int(y1)
                            #    w1 = int(w1)
                            #    h1 = int(h1)
                            #    x2 = int(x2)
                            #    y2 = int(y2)
                            #    w2 = int(w2)
                            #    h2 = int(h2)
                            #    asdasf = originalFrame.copy()
                            #    cv2.rectangle(asdasf, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 1)
                            #    cv2.rectangle(asdasf, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 1)
                            #    asdasf = cv2.resize(asdasf, (asdasf.shape[1]//2, asdasf.shape[0]//2))
                            #    cv2.imshow("YOLOv8 Tracking", asdasf)
                            #    cv2.waitKey(0)
                            #    pass
                            #list_a.append(float(IoU))            


    # Tenim valors a eliminar repetits molts cops
    # ens quedem nomes amb 1 de cada
    print("Valors a eliminar: ", len(to_remove))
    unique_set = set(tuple(sublist) for sublist in to_remove)
    to_remove = [list(item) for item in unique_set]
    print("Valors a eliminar: ", len(to_remove))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################
    ###  Elimino aquells BBoxes que he vist que se solapen  ###
    ###########################################################
    
    to_remove_definitiu = []
    for index, MOT in enumerate(frameMOT):
        for remove in to_remove:
            if MOT == remove:
                to_remove_definitiu.append(index)
                break




    to_remove_definitiu = list(set(to_remove_definitiu))
    to_remove_definitiu.sort(reverse = True) 
    for i in to_remove_definitiu:
        frameMOT.pop(i)    

    #for bbox in frameMOT:
    #    x, y, w, h = bbox[2:]
    #    x = int(x)
    #    y = int(y)
    #    w = int(w)
    #    h = int(h)
    #    #cv2.rectangle(originalFrame, (x+ random.randint(1, 20), y+ random.randint(1, 20)), (x+ random.randint(1, 20) + w, y+ random.randint(1, 20) + h), (0, 0, 255), 1)
    #    #cv2.rectangle(originalFrame, (x+ random.randint(1, 20), y+ random.randint(1, 20)), (x+ random.randint(1, 20) + w, y+ random.randint(1, 20) + h), (0, 0, 0), 1)
    #    cv2.rectangle(originalFrame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    #originalFrame = originalFrame[300:700, 3000:3600]
    ##resized_frame = cv2.resize(originalFrame, (frame.shape[1]//2, frame.shape[0]//2))
    #cv2.imshow("YOLOv8 Tracking", originalFrame)        
    #cv2.waitKey(0)

    return frameMOT



