# FIRST Robotics Competition Data Challenge
Data Science Capstone project for spring 2026. \
**Team Members:** Kathryn Keck, Kennedy Wendl, Kylie Tauke,​ Ryan Bogdan, and Trae Anderson

## Project Description
The Blue Alliance (TBA) has data for every FIRST Robotics competition, a worldwide robotics competition centered around teams of high school students who, with the assistance of mentors, engineer a single robot to compete in a game as outlined in the beginning of the season. These robots then travel for competitions, where they play multiple matches. For each match, robots get groups into teams of three called alliances, and two alliances are then pitted against each other. The data TBA provides is broken down by alliance for every match. With the data broken down to the alliance level, it can be difficult to scout specific robots and identify which ones are powerhouses versus being carried by the other robots in their alliance. Our goal for our project was to use computer vision to process the videos from the season of 2025 provided by TBA, break down the scoring by robot for each match rather than by alliance, and deliver this data (and our code) to our sponsor, streamlining robot scouting.

## Branch Overview 
**Main:** Contains finalized work and results of the project. \
**Workflow:** Contains work completed by the team throughout the semester. Some folders/files within this branch may be unfinished, and should only be used to reference our past efforts!

## Setup and Execution
**Environment Notes**
Our project can be run on macOS or Windows. Both train.py and run.py are designed to run on GPU clusters. By default, DEVICE = 0 targets the first available GPU. However this can be changed to "cpu" for CPU-only execution but it is significantly slower! 

### 1. Training Setup 
```bash
# Install all Python dependencies before running 
pip3 install ultralytics torch pandas easyocr opencv-python yt-dlp imageio-ffmpeg matplotlib numpy

# Install necessary Node modules (only required for the first time!)
npm install

# Run the training script
python train.py 
```

### 2. Results Setup 
```bash
# Run main pipeline 
python run.py 
```

## Model Training
This section explains how we created our models that allowed us to get scoring data by robot!

### Roboflow (Training)
To create training data for our YOLO model, which is used for object identification, we utilized Roboflow. Roboflow is a free-to-use app that allowed us to take frames from competition videos and draw boxes around the reef, robots, and scored coral to classify them as their particular type of object. 
We also used Roboflow to assist in reading the scoreboard, particularly to retain the time on the clock, red and blue alliance scores, and red and blue alliance robots.

### YOLO Model (Identification)
We used our Roboflow robot training frames to train our YOLO robot model, which, after training, is able to take new videos, break them down into frames, and identify the objects in the frames that we classified in the training data. METRICS TO SUMMARIZE HOW WELL IT DID?
We did the same for our YOLO scoreboard model, which could then extract the items described above from any frame. We combined that with EasyOCR, a text recognition program, so that we could have a CSV that contains an observation for every time the score changed. Each observation includes the red and blue alliance scores, the time on the clock when the score changed, the lists of robots in the red and blue alliances, and the YouTube link of the match. We also added the side of the screen the red score was on, since that aligns with the side the red reef is on, along with a flag that indicates whether the match is in the 'auto' phase or the 'teleop' phase. MORE METRICS TO SUMMARIZE?

### BoT-SORT Model (Tracking)
After identification, the next step was training our BoT-SORT model.

### Game Logic Implementation (Scoring)
Finally, once our Botsort model was trained, we could implement the scoring logic to create the scoring data by robot.

## Processing Pipeline
This section describes how to use our code on new videos to generate robot-level data!

### Video Processing
Our video processing script takes a video and crops it to the lower view, which includes the views of both of the reefs. This prevents our models from getting confused by people, technology, etc. that are in the top view. After this, the cropped video can be plugged into our models to be evaluated into data.

