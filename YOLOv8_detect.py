from ultralytics import YOLO
import os
import random
import cv2
import numpy as np

DIR = r"datasets\test\ant_subset_1-000_det_dataset_0_000226_1_7.png" # We can use a single image, a folder with images or a video


#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi


img_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp",".BMP"]
vid_extensions = [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".wmv", ".WMV"]


if os.path.isdir(DIR):
    FILE = "IMGS"
elif os.path.isfile(DIR):
    if DIR.endswith(tuple(img_extensions)):
        FILE = "IMG"
    elif DIR.endswith(tuple(vid_extensions)):
        FILE = "VID"
    else:
        print("File format not supported")






## If data is an image
if FILE == "IMG":
    frame = cv2.imread(DIR)

## If data is a group of images
if FILE == "IMGS":
    frame = []
    fileNames = []
    directory_out = os.path.join(DIR, 'out')
    if not os.path.exists(directory_out):
        os.makedirs(directory_out)

    for f in os.listdir(DIR):
        if f.endswith(tuple(img_extensions)):
            if os.path.isfile(os.path.join(DIR, f)):
                frame.append(cv2.imread(os.path.join(DIR, f)))

                fileNames.append(f)

## If data is a video                    
if FILE == "VID":
    
    directory_out = DIR.split(".")[0]+"_out.mp4"
    #directory_out = DIR
    #directory_out = directory_out.replace('.avi', '_out.avi')

    cap = cv2.VideoCapture(DIR)
    print(directory_out)
    ret, frame = cap.read()
    cap_out = cv2.VideoWriter(directory_out, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                              (frame.shape[1], frame.shape[0]))
    
    frame_cnt = 0




# Load the model
model = YOLO(r"models_detect_pol\best.pt")




## If data is an image
if FILE == "IMG":
    results = model(frame)

    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > 0.5:
                detections.append([class_id, x1, y1, x2, y2, score, result.names[class_id]])

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(len(detections))]    
    for i in detections:
        #print(i)
        frame = cv2.rectangle(frame, (i[1], i[2]), (i[3], i[4]), colors[i[0]%len(detections)], 2)
        frame = cv2.putText(frame, i[-1], (i[1], i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i[0]%len(detections)], 2)
        frame = cv2.putText(frame, str(np.round(i[-2],1)), (i[1], i[2]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i[0]%len(detections)], 2)

## If data is a group of images
if FILE == "IMGS":
    for index, f in enumerate(frame):
        results = model(f)

        detections = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > 0.5:
                    detections.append([class_id, x1, y1, x2, y2, score, result.names[class_id]])

        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(len(detections))]    
        for i in detections:
            #print(i)
            f = cv2.rectangle(f, (i[1], i[2]), (i[3], i[4]), colors[i[0]%len(detections)], 2)
            f = cv2.putText(f, i[-1], (i[1], i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i[0]%len(detections)], 2)
            f = cv2.putText(f, str(np.round(i[-2],1)), (i[1], i[2]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i[0]%len(detections)], 2)     
        
        cv2.imwrite(os.path.join(directory_out, "out_"+fileNames[index]), f)

## If data is a video
if FILE == "VID":    
    while ret and frame_cnt < 200:
        results = model(frame)    
    
        detections = []
        for result in results:
    
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > 0.1:
                    detections.append([class_id, x1, y1, x2, y2, score, result.names[class_id]])
    
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(len(detections))]    
        for i in detections:
            #print(i)
            frame = cv2.rectangle(frame, (i[1], i[2]), (i[3], i[4]), colors[i[0]%len(detections)], 2)
            frame = cv2.putText(frame, i[-1], (i[1], i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i[0]%len(detections)], 2)
            frame = cv2.putText(frame, str(np.round(i[-2],1)), (i[1], i[2]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i[0]%len(detections)], 2)
    
    
        cap_out.write(frame)
        ret, frame = cap.read()
        frame_cnt += 1






## If data is an image
if FILE == "IMG":
    cv2.imshow('image', frame)

## If data is a group of images
if FILE == "IMGS":
    os.open(directory_out)

## If data is a video
if FILE == "VID":
    cap.release()
    cap_out.release()


cv2.waitKey(0)
cv2.destroyAllWindows()


    
