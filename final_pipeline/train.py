# imports
from ultralytics import YOLO

# train robot YOLO
model = YOLO("training/yolov8s.pt")

results = model.train(
    data="training/robot_model/yolov8_training_images/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)