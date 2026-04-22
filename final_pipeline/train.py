# Imports
from ultralytics import YOLO

# Yolo model used for training the coral, robots, team numbers, and scoreboard! 
model = YOLO("training/yolov8m.pt")

# Training and finding the best YoloV8 model for robot detection. 
# We decided to focus tuning only on the robots however similar logic can be applied for tuning the coral, team number and scoreboard models!
# YoloV8 has great documentation on hyperparameter tuning and can be found here to be apply to other models: https://docs.ultralytics.com/usage/cfg/#modes 

# The parameters provided are default but these are later changed in Line 73. 
def tune_yolov8_for_robots(model_path="yolov8m.pt", data="training/robot_model/data.yaml",
        epoch_list=[20, 50, 100],
        lr_list=[0.01, 0.005, 0.001],
        batch_list=[8, 16],
        imgsz=640):

    results_summary = []

    best_map = 0
    best_model_path = None

    for epochs, lr, batch in itertools.product(epoch_list, lr_list, batch_list):

        # Statement to print status of training
        # print(f"\nTraining with epochs={epochs}, lr={lr}, batch={batch}")

        model = YOLO(model_path)

        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=0,
            optimizer="SGD",
            lr0=lr,
            momentum=0.937,
            weight_decay=0.0005
        )

        metrics = model.val()
        map5095 = metrics.box.map

        results_summary.append({
            "epochs": epochs,
            "lr": lr,
            "batch": batch,
            "mAP50": metrics.box.map50,
            "mAP50-95": map5095
        })

        # Check if model is best 
        # Best model is determined by map5095 score
        if map5095 > best_map:
            best_map = map5095

            # Grabs path of 
            best_weights = Path(results.save_dir) / "models" / "weights" / "best.pt"
            best_model_path = Path("best_tuned_yolov8.pt")

            shutil.copy(best_weights, best_model_path)

            print(f"Bst model saved with mAP50-95 value of {best_map:.4f}.")


    df = pd.DataFrame(results_summary)
    print(f"\nBest model saved at: {best_model_path}")

    return df

# Saving best model, given the below paarameters, for the robot model. These values can be changed to see what achieves the highest accuracy in identifying robots. 
# epoch_list: Forward and backward pass of training data, impacts the model’s ability to find patterns. We kept these values low because a high epoch size can lead to overfitting and without a GPU takes a long time to run!
# lr_list: Learning rate, impacts how fast the model learns.
# batch_list: Number of images processed simultaneously. The smaller the batch, the more at risk you are for noise.
tune_yolov8(
    epoch_list=[20, 50],           
    lr_list=[0.01, 0.005, 0.001],
    batch_list=[8, 16]
)

# Training the model that will crop the scoreboard from the video 
model.train(
    data="training/crop_scoreboard/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    project="models/crop_scoreboard",
    name="crop_scoreboard"
)

# Training the model that will find the scoreboard info (blue and red team scores, and time remaining)
model.train(
    data="training/extract_scoreboard_info/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    project="models/extract_scorboard",
    name="extract_scoreboard"
)

# Training the model to identify team numbers 
model.train(
    data="training/robot_numbers/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="models/best_number",
    name="best_number"
)

# Training the model to identify coral 
model.train(
    data="training/coral_model/data.yaml",
    epochs=100,        
    imgsz=640,        
    batch=16,  
    project="models/best_coral",        
    name='best_coral'
)
