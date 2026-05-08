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
Our project can be run on macOS or Windows. Both train.py and run.py are designed to run on GPU clusters. By default, DEVICE = 0 targets the first available GPU. However, this can be changed to "cpu" for CPU-only execution, but it is significantly slower! 

The following commands must be run in the terminal to view our project. Depending on your system's version of Python, "pip" and "python" may need to be changed to "pip3" or "python3". 

### 1. Training Setup 
```bash
# Install all Python dependencies before running 
pip install ultralytics torch pandas easyocr opencv-python yt-dlp imageio-ffmpeg matplotlib numpy

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

### 3. Video Processing
Once run.py has begun processing, the user will be prompted to enter a YouTube link. Here, you will type or paste the link to your desired match from the 2025 FIRST Robotics Competition season into the terminal!

### 4. Robot Scoring with the Terminal 
After the YouTube link has been entered, the user may be prompted to identify the robot that most recently scored on the L4 branch. Our program will prompt the user, from the terminal, to enter a number 0 - 6. Each number corresponds to a team number from the robots competing in the match, with 0 indicating an unknown team. Our program asks the user to manually select a robot's team number when it can not identify the robot from the video. 

<img width="610" height="383" alt="image" src="https://github.com/user-attachments/assets/139c5d8e-49b2-41d5-b24d-1dd97ab27cd7" /> \ 

The photo above is an example of a scoring incident image that would be added to the output/scoring/ folder! The name of the photo (EX: event_0_frame_294_curr_nan_closest_10.png) will be given from the terminal. The user must then manually open the photo to identify the robot surrounded by the green frame. The green frame highlights which robot our model believed to score a point, and is what team number (0 - 6) the user should enter into the terminal! This photo may also contain robots surrounded by red and blue frames, but the user should ignore the other robots and only enter the number that corresponds to the team number of the robot highlighted in the green.  

## Model Training
How we created the models that allowed us to get scoring data from the robot.

### Roboflow (Training)
To create training data for our YOLO model used for object identification, we used Roboflow. Roboflow is a free-to-use website that lets us upload frames from competition videos and draw boxes around the reef, robots, and scored coral to classify them as their respective object types. 
We also used Roboflow to draw boxes around the sections of the scoreboard we used for our code.

### YOLO Model (Identification)
We used our Roboflow robot training frames to train our YOLO robot model, which, after training, is able to take new videos, break them down into frames, and identify the reefs, robots, and coral in the frames that we classified in the training data.
We did the same for our YOLO scoreboard model, which could then extract the items described above from any frame. We combined that with EasyOCR, a text recognition program, so that we could have a CSV that contains an observation for every time the score changed. Each observation includes the red and blue alliance scores, the time on the clock when the score changed, the lists of robots in the red and blue alliances, and the YouTube link of the match. We also added the side of the screen on which the red score was, since that aligns with the side on which the red reef is, along with a flag that indicates whether the match is in the 'auto' phase or the 'teleop' phase.

### BoT-SORT Model (Tracking)
After identification, we use Bot-SORT to track the robots through a match. This allows us to track the robot frame-by-frame in a video. If the robot moves out of frame or behind the reef and the model loses it, the robot will get a new ID. We then try to read the number off the robot bumper at some point during the track. We give them a list of robots to choose from in that match. If we can identify the robot number, we know which robot it is for the entire track.

### Game Logic Implementation (Scoring)
We combine our previous models to calculate scoring. We focus only on the L4 coral scored on the reef. We utilize a Markov model with our YOLO identification to
decide when a coral has been scored on the L4 section of the reef. When a coral has been scored, we look for the nearest robot to the section that was scored and
attribute a score to that robot. If we have identified the robot, we can assign a score to it; otherwise, we ask the user to identify which robot it is.

### Video Processing
Our video processing script takes a video and crops it to a lower view that shows both reefs. This prevents our models from getting confused by people, technology, etc., that are in the top view. After this, the cropped video can be plugged into our models for evaluation as data.

