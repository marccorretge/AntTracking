from ultralytics import YOLO
from splitFrames import splitFrame, joinROIs
import cv2
import numpy as np
import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('YOLO_LETTER', type=str, help='name YOLO model: n (YOLOv8n)')
parser.add_argument('YOLO_DATASET', type=str, help='Pol, Marc, PolMarc...')

args = parser.parse_args()

YOLO_LETTER = args.YOLO_LETTER
YOLO_DATASET = args.YOLO_DATASET

ROI = 640
OVERLAP = 0.2

DETECT = True
TRACK = False

#model = YOLO("yolov8n.pt")  # Load an official Detect model
model = YOLO("/mnt/work/users/marc.corretge/modelsEntrenats/YOLOv8" + YOLO_LETTER + "_best_" + YOLO_DATASET + ".pt")


# Open the video file
video_path = "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps = cap.get(cv2.CAP_PROP_FPS)

newFolder = "resultsEvalPerMemoria_iouSplit-0.75"
newOutPath = os.path.join("/home/usuaris/imatge/marc.corretge/AntTracking/YOLO_TRACKING/", newFolder)
os.makedirs(newOutPath, exist_ok=True)



#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi


# Define the codec and create VideoWriter object
#output_path = "E:/TFG/dataset/all_ants/all_ants_0-063_annotatedWithYOLO_output.mp4"
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

# Store the track history
track_history = defaultdict(lambda: [])


frameCount = 0

MOT = []
success = True

while success:        
    print("Frame: ", frameCount, flush=True)   

    # Read a frame from the video


    success, frame = cap.read()  
    if success:           
        frame_backup = frame.copy() 
        frameCount += 1 
        
        frameMOT = []

        # Split the frame into regions of interest (ROIs)
        if frame.shape[0] == ROI and frame.shape[1] == ROI:
            print("ROI and frame shape are equals: overlap = 0", flush=True)
            roi_frame = splitFrame(frame, ROI, overlap=0.0)
        else:
            roi_frame = splitFrame(frame, ROI, overlap=OVERLAP)
        
        roi_dict = {}

        for key, roi in roi_frame.items():
            print(key)
            #cv2.rectangle(frame, (key[1], key[2]), (key[1] + ROI, key[2] + ROI), (255, 0, 0), 2)
            #frame_resized = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
            #cv2.imshow("frame_resized", frame_resized)
            #print("ROI: ", key)
            #cv2.waitKey(0)

            # Function to convert RGB to grayscale with 3 channels
            # Same results using RGB or grayscale images
            def rgb_to_gray_3_channels(image):
                return image
                gray_single_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_3_channel = cv2.merge([gray_single_channel, gray_single_channel, gray_single_channel])
                return gray_3_channel
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            if TRACK:
                results = model.track(rgb_to_gray_3_channels(roi), persist=False)
            if DETECT:
                results = model(rgb_to_gray_3_channels(roi))

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            scores = results[0].boxes.conf

            # En cas de que no hi hagi cap detecció, donarà error
            if TRACK:
                try:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                except:
                    continue
                    # Evitem que el programa peti si no hi ha cap detecció
                    pass

            # Visualitzc the results on the frame
            annotated_frame = results[0].plot()
            scores = results[0].boxes.conf

            # Save to MOT   
            if TRACK:         
                for index, box, track_id in enumerate(zip(boxes, track_ids)):
                    cx, cy, w, h = box
                    cx = cx + key[1]
                    cy = cy + key[2]

                    x0 = cx - w/2
                    y0 = cy - h/2
                    
                    score = float(scores[index])
                    # if one detection is touching the border of the frame, we ignore it
                    if np.round(x0) <= 0 + key[1] or np.round(y0) <= 0 + key[2] or np.round(x0 + w) >= ROI + key[1] or np.round(y0 + h) >= ROI + key[2]:
                        continue
                    else:
                        frameMOT.append([frameCount, track_id, x0, y0, w, h, score])
                    track = track_history[track_id]
                    track.append((float(cx), float(cy)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
            
            if DETECT:
                for index, box in enumerate(boxes):
                    cx, cy, w, h = box
                    cx = cx + key[1]
                    cy = cy + key[2]

                    x0 = cx - w/2
                    y0 = cy - h/2
                    
                    score = float(scores[index])

                    if x0 + w > frame.shape[1] or y0 + h > frame.shape[0]:
                        continue
                    #if x0 > 3000:
                    #cv2.rectangle(frame_backup, (int(x0), int(y0)), (int(x0) + int(w), int(y0) + int(h)), (0, 0, 0), 3)
                    #cv2.putText(frame_backup, str(score), (int(x0), int(y0 + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    # draw rectangle on the frame from key[1] to key[1] + key[3] and key[2] to key[2] + key[4]
                    #cv2.rectangle(frame, (key[1], key[2]), (key[1] + 640, key[2] + 640), (255, 0, 0), 2)
                    
                    
                    
                    # if one detection is touching the border of the frame, we ignore it
                    if np.round(x0) <= 0 + key[1] or np.round(y0) <= 0 + key[2] or np.round(x0 + w) >= ROI + key[1] or np.round(y0 + h) >= ROI + key[2]:
                        #cv2.rectangle(frame_backup, (int(x0), int(y0)), (int(x0) + int(w), int(y0) + int(h)), (0, 255, 0), 4)
                        #cv2.putText(frame_backup, str(int(score*100)), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
                        #print("no ", x0, y0, w, h, score, flush=True)
                    #if np.round(x0) <= 0 or np.round(y0) <= 0 or np.round(x0 + w) >= ROI or np.round(y0 + h) >= ROI:
                        continue
                        #frameMOT.append([frameCount, -1, float(x0), float(y0), float(w), float(h), score])
                    else:
                        #cv2.rectangle(frame_backup, (int(x0), int(y0)), (int(x0) + int(w), int(y0) + int(h)), (0, 0, 0), 2)
                        #cv2.putText(frame_backup, str(int(score*100)), (int(x0), int(y0 + 100)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
                        frameMOT.append([frameCount, -1, x0, y0, w, h, score])
                        #print("si ", x0, y0, w, h, score, flush=True)

                    frame_backup_resized = cv2.resize(frame_backup, (frame.shape[1]//2, frame.shape[0]//2))
                    #cv2.imshow("YOLOv8 Tracking",frame_backup_resized)
                    #cv2.waitKey(15)
            # Write the annotated frame to the output video
            #out.write(annotated_frame)
            #if key == (7, 512, 512):
            #annotated_frame_resized = cv2.resize(annotated_frame, (annotated_frame.shape[1]//2, annotated_frame.shape[0]//2))
            #cv2.imshow("YOLOv8 Tracking", annotated_frame_resized)
            #cv2.waitKey(0)
            pass

            roi_dict[key] = annotated_frame

            # Display the annotated frame (optional)
            #cv2.imshow("YOLOv8 Tracking", annotated_frame)
            #pass

            # Break the loop if 'q' is pressed (optional)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        # Join the ROIs in to a single frame
       
        

        frameMOT = joinROIs(frame, roi_dict, frameMOT, frame_width, frame_height)   

        for line in frameMOT:
            x0 = int(line[2])
            y0 = int(line[3])
            w = int(line[4])
            h = int(line[5])
            score = np.round(line[6],3)
            #cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 2)
            #cv2.putText(frame, str(score), (x0, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            pass
            
        #frame_resized = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        #cv2.imshow("YOLOv8 Tracking - after reJoint", frame_resized)
        #print("Frame: ", frameCount, flush=True)
        #cv2.waitKey(0)
        MOT.append(frameMOT)     
    else:
        # Acabem el loop quan s'acaba el video
        break

# Finish the video writing
cap.release()


# save MOT in txt file
with open(os.path.join(newOutPath, "/ant_subset_1-024_YOLOv8" + YOLO_LETTER + "_" + YOLO_DATASET+"_MOT.txt"), "w") as f:
    for frame in MOT:
        for row in frame:
            row = np.array(row)
            f.write(str(int(row[0])) + ",-1," + str(np.round(row[2],5)) + "," + str(np.round(row[3],5)) + "," + str(np.round(row[4],5)) + "," + str(np.round(row[5],5)) + "," + str(np.round(row[6],5)) + ",-1,-1,-1\n")
    pass



