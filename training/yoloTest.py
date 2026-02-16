from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight, fast

model.predict(
    source="testMatch.mp4",   # your video
    conf=0.4,
    save=True
)