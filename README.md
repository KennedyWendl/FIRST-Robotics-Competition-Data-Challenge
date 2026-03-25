# FIRST-Robotics-Competition-Data-Challenge
Data Science Capstone project for spring 2026. 

The Blue Alliance (TBA) has data for every FIRST Robotics Competition. Scoring is broken down by alliance for every match. There are three robots per alliance. Our goal was to process the videos provided by TBA to break down the scoring by robot for each match, streamlining robot scouting.

## Model Training
How we created our models that allowed us to get scoring data by robot!

### Roboflow (Training)
To create training data for our YOLO model, which is used for object identification, we utilized Roboflow. Roboflow is a free-to-use app that allowed us to take frames from competition videos and draw boxes around the reef, robots, and scored coral to classify them as their particular type of object.

### YOLO Model (Identification)
We used our Roboflow training frames to train our YOLO model, which, after training, is able to take videos and identify the objects we classified in the training data.


### Botsort Model (Tracking)
After identification, the next step was training our Botsort model.

### Game Logic Implementation (Scoring)
Finally, once our Botsort model was trained, we could implement the scoring logic to create the scoring data by robot.

## Processing Pipeline
How to use our code on new videos!

### Video Processing
Our video processing script takes a video and crops it to the lower view, which includes the views of both of the reefs. This prevents our models from getting confused by people, technology, etc. that are in the top view. After this, the cropped video can be plugged into our models to be evaluated into data.

### Data Output

### Dashboard Output