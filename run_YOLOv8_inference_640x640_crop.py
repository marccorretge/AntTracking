from ultralytics import YOLO
from splitFrames import splitFrame, joinROIs
import argparse
import cv2
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('YOLO_LETTER', type=str, help='name YOLO model: n (YOLOv8n)')
parser.add_argument('YOLO_DATASET', type=str, help='Pol, Marc, PolMarc...')

args = parser.parse_args()

YOLO_LETTER = args.YOLO_LETTER
YOLO_DATASET = args.YOLO_DATASET


model = YOLO("/mnt/work/users/marc.corretge/modelsEntrenats/YOLOv8" + YOLO_LETTER + "_best_" + YOLO_DATASET + ".pt")

video_path = "/mnt/work/datasets/AntTracking/videos/ant_subset_1/ant_subset_1-024.mp4"
cap = cv2.VideoCapture(video_path)


TRACK = True
DETECT = False

ROI = 640
xCrop = 350
yCrop = 1750


#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi



import ctypes
# Obtener las dimensiones de la pantalla
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Calcular la posición para centrar la ventana
window_width = ROI
window_height = ROI
xpos = (screen_width - window_width) // 2
ypos = (screen_height - window_height) // 2

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
#output_path = "./all_ants_0-063_annotatedWithYOLO_output.mp4"
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
##out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

# Store the track history
track_history = defaultdict(lambda: [])



cont = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:
        cont += 1
        # crop frame in ROIxROI
        frame = frame[yCrop:yCrop+ROI, xCrop:xCrop+ROI]
        #cv2.imshow("YOLOv8 Tracking", frame)
        pass

        if TRACK:
            results = model.track(frame, persist=True)
        if DETECT:
            results = model(frame)


        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        if TRACK:        
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        if TRACK:
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                centerX = int(x)
                centerY = int(y)
                w = int(w)
                h = int(h)
                x = centerX - w // 2
                y = centerY - h // 2
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

                track = track_history[track_id]
                track.append((float(centerX), float(centerY)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 0), thickness=2)

        if DETECT:
            for box in boxes:
                x, y, w, h = box
                centerX = int(x)
                centerY = int(y)
                w = int(w)
                h = int(h)
                x = centerX - w // 2
                y = centerY - h // 2
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        
        # Display the annotated frame (optional)
        cv2.imshow("Real Time YOLOv8 Tracking", annotated_frame)
        cv2.waitKey(15)
        # Mover la ventana al centro de la pantalla
        cv2.moveWindow("Real Time YOLOv8 Tracking", xpos, ypos)
        pass

        # Break the loop if 'q' is pressed (optional)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects and close the display window
cap.release()
#out.release()
cv2.destroyAllWindows()