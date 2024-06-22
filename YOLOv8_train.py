from ultralytics import YOLO
 

YOLO_LETTER = "n"
YAML_PATH = "/mnt/work/users/marc.corretge/train_YOLO_imatgesMeves/config.yaml"
IMGSZ = 640
EPOCHS = 100
BATCH = 16
OUT_NAME = "yolov8n_Marc"

#  Final declaracio de variables
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#  Inici del codi



# Load the model.
model = YOLO("yolov8" + YOLO_LETTER + ".pt")

# Training.
results = model.train(
   data = YAML_PATH,
   imgsz = IMGSZ,
   epochs = EPOCHS,
   batch = BATCH,
   name = OUT_NAME)


print(model.metrics)