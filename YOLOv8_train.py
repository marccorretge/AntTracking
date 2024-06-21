from ultralytics import YOLO
 
# Load the model.
model = YOLO("yolov8n.pt")

# Training.
results = model.train(
   data = "/mnt/work/users/marc.corretge/train_YOLO_imatgesMeves/config.yaml",
   imgsz=640,
   epochs=100,
   batch=16,
   name='yolov8n_Marc')


print(model.metrics)