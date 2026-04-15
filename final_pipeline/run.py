
# Temporary Outline for how the code will flow
# 
# 1. YT Link Provided
# 2. Download YT Video
# 3. Run Scoreboard model
# 4. Run video cropping
# 5. If robot path is wanted, have point selection for homography happen
# 6. Identify Robots
# 7. Track Robots
# 8. Output Stuff

import os
import re
import cv2
import pandas as pd
from yt_dlp import YoutubeDL
import imageio_ffmpeg
from ultralytics import YOLO
import easyocr
import numpy as np
from collections import defaultdict, Counter


# --- User Settings --- 

# The youtube video that will be processed
YOUTUBE_LINK = "https://www.youtube.com/watch?v=NSWVoO4ZDEs" 

# This variable specifies what GPU(s) you use (if available). Can be set to "cpu", 0, [0,1], etc.
DEVICE = 0  

# Set to True if you want an image of the robot paths. Set to False otherwise. If running on a server, you'll need to set to False since it requires you to select points to perform homography.
OUTPUT_ROBOT_PATHS = True

# The path of the YOLOv8 model that is used to crop the scoreboard from the raw videos.
CROP_SCOREBOARD_MODEL_PATH = "models/crop_scoreboard.pt" 

# The path of the YOLOv8 model that extracts the information from the cropped scoreboard.
SCOREBOARD_INFO_MODEL_PATH = "models/extract_scoreboard_info.pt" 

# How many frames are skipped between each processing step for the scoreboard. If the value is 15, it will process 1 frame every 15 frames.
SCOREBOARD_FRAME_SKIP = 15 

# Whether to delete the downloaded video after processing to save space. Set to True to enable deletion.
DELETE_VIDEO = False 



# Initialize directories and models
os.makedirs("videos", exist_ok=True)

crop_model = YOLO(CROP_SCOREBOARD_MODEL_PATH)
info_model = YOLO(SCOREBOARD_INFO_MODEL_PATH)

reader = easyocr.Reader(['en'], gpu=(DEVICE != "cpu"))

# Scoreboard: Used to enhance the region of interest (ROI) for better OCR performance. It resizes, increases contrast, and applies thresholding.
def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=10)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh_inv = cv2.bitwise_not(thresh)
    return [thresh, thresh_inv]

# Scoreboard: Runs OCR on multiple images and returns the highest-confidence text
def read_best_text(images, allowlist):
    best_text, best_conf = None, 0

    for img in images:
        results = reader.readtext(
            img,
            allowlist=allowlist,
            detail=1,
            paragraph=False
        )
        for (_, text, conf) in results:
            if conf > best_conf:
                best_conf, best_text = conf, text

    return best_text

# Scoreboard: Extracts the numeric value from the image using OCR
def read_number(img):
    images = preprocess_for_ocr(img)
    text = read_best_text(images, '0123456789')

    if text is None:
        return None

    text = re.sub(r"\D", "", text)
    return int(text) if text else None

# Scoreboard: Extracts the timer value in MM:SS format from the image using OCR
def read_timer(img):
    images = preprocess_for_ocr(img)
    text = read_best_text(images, '0123456789:')

    if text is None:
        return None

    match = re.search(r"\d{1,2}:\d{2}", text)
    return match.group(0) if match else None

# Download video at best quality and return the saved filename
def download_video(url, output_dir):
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "merge_output_format": "mp4",
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return os.path.splitext(os.path.basename(filename))[0] + ".mp4"

# Download the video
video_filename = download_video(YOUTUBE_LINK, "videos")
video_path = os.path.join("videos", video_filename)

# Storage for final rows and tracking previous scores
rows = []
prev_blue = None
prev_red = None
pending_row = None # Holds the first occurence of a score change when the timer is missing
is_auto = True

# Extract the youtube url for this video
youtube_url = YOUTUBE_LINK

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames according to FRAME_SKIP
    if frame_idx % SCOREBOARD_FRAME_SKIP != 0:
        frame_idx += 1
        continue
    
    # Detect the scoreboard region
    crop_results = crop_model(frame, device=DEVICE)[0]

    # Skip if no scoreboard is found
    if len(crop_results.boxes) == 0:
        frame_idx += 1
        continue

    # Crop scoreboard from frame
    x1, y1, x2, y2 = map(int, crop_results.boxes.xyxy[0])
    scoreboard = frame[y1:y2, x1:x2]

    # Find the elements inside the cropped frame (scores, timer, team numbers)
    info_results = info_model(scoreboard, device=DEVICE)[0]

    # Initialize variables for this frame
    blue_score = None
    red_score = None
    timer = None


    blue_center_x = None
    red_center_x = None

    team_data = []

    # Loop through each of the detected elements
    for b in info_results.boxes:

        cls_id = int(b.cls[0])
        label = info_model.names[cls_id]

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        region = scoreboard[y1:y2, x1:x2]

        x_center = (x1 + x2) / 2

        # Read blue score
        if label == "blue_score":
            blue_score = read_number(region)
            blue_center_x = x_center

        # Read red score
        elif label == "red_score":
            red_score = read_number(region)
            red_center_x = x_center

        # Read timer
        elif label == "timer":
            timer = read_timer(region)

        # Read team number
        elif label == "team_number":
            num = read_number(region)
            if num is not None:
                team_data.append((num, x_center))

    # If both scores and timer are missing, skip the frame
    if blue_score is None and red_score is None and timer is None:
        frame_idx += 1
        continue

    # Ensure blue score doesn't decrease compared to previous frames
    if prev_blue is not None and blue_score is not None:
        if blue_score < prev_blue:
            frame_idx += 1
            continue

    # Ensure red score doesn't decrease compared to previous frames
    if prev_red is not None and red_score is not None:
        if red_score < prev_red:
            frame_idx += 1
            continue

    # If the scores haven't changed, skip the frame
    if prev_blue == blue_score and prev_red == red_score:
        frame_idx += 1
        continue

    # Determine if it is still auto stage or not
    if timer is not None and is_auto:
        try:
            minutes = int(timer.split(":")[0])
            if minutes == 2:
                is_auto = False
        except:
            pass

    # Identify if the red score is on the left or right side of the scoreboard
    red_location = None
    if blue_center_x is not None and red_center_x is not None:
        red_location = "right" if red_center_x > blue_center_x else "left"

    # Split the team numbers into blue vs red based on their loation to the score
    blue_teams = []
    red_teams = []

    midpoint = (blue_center_x + red_center_x) / 2 if blue_center_x and red_center_x else None

    if midpoint:
        for num, x in team_data:
            if red_location == "right":
                if x < midpoint:
                    blue_teams.append((num, abs(x - blue_center_x)))
                else:
                    red_teams.append((num, abs(x - red_center_x)))
            else:
                if x > midpoint:
                    blue_teams.append((num, abs(x - blue_center_x)))
                else:
                    red_teams.append((num, abs(x - red_center_x)))

    # Keep the closest 3 teams for each alliance
    blue_team_numbers = [n for n, _ in sorted(blue_teams, key=lambda x: x[1])[:3]]
    red_team_numbers = [n for n, _ in sorted(red_teams, key=lambda x: x[1])[:3]]

    # Build row
    current_row = {
        "Frame": frame_idx,
        "blue_score": blue_score,
        "red_score": red_score,
        "timer": timer,
        "is_auto": is_auto,
        "red_location": red_location,
        "blue_team_numbers": blue_team_numbers,
        "red_team_numbers": red_team_numbers,
        "youtube_link": youtube_url,
    }

    # If the timer is missing, store the very first instance for this score change
    if blue_score is not None and red_score is not None and timer is None:
        if pending_row is None:
            pending_row = current_row
        frame_idx += 1
        continue

    # If the timer ends up appearing in a later frame with the same scores, use that row instead
    if (
        pending_row is not None and
        timer is not None and
        blue_score == pending_row["blue_score"] and
        red_score == pending_row["red_score"]
    ):
        rows.append(current_row)
        pending_row = None

    # If score changes, flush the pending row and add the current row
    else:
        if pending_row is not None:
            rows.append(pending_row)
            pending_row = None

        rows.append(current_row)

    # Update previous scores
    prev_blue = blue_score
    prev_red = red_score

    frame_idx += 1

# Save any remaining pending row at the end of the video
if pending_row is not None:
    rows.append(pending_row)
    pending_row = None

cap.release()

# Delete video after processing to save space
if DELETE_VIDEO == True and os.path.exists(video_path):
    os.remove(video_path)

# Convert to dataframe and display
scoreboard_df = pd.DataFrame(rows)


