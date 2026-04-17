# imports
from ultralytics import YOLO

# the YOLO models that are used for training
robot_model = YOLO("training/yolov8s.pt")
crop_scoreboard_model = YOLO("training/yolov8m.pt")
extract_scoreboard_model = YOLO("training/yolov8m.pt")

# training the robot model
robot_results = robot_model.train(
    data="training/robot_model/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

# training the crop scoreboard model
crop_scoreboard_model.train(
    data="training/crop_scoreboard/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    workers=4
)

# training the extract scoreboard info model
extract_scoreboard_model.train(
    data="training/extract_scoreboard_info/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=4
)

