from ultralytics import YOLO
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

model = YOLO("yolov8m.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=4
)