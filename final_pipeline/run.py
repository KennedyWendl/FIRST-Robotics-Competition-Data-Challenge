
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
import math
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


# --- User Settings --- 

# The youtube video that will be processed
YOUTUBE_LINK = "https://www.youtube.com/watch?v=NSWVoO4ZDEs" 

# This variable specifies what GPU(s) you use (if available). Can be set to "cpu", 0, [0,1], etc.
DEVICE = 0  

# If script is run with command line argumment --robot_path it will output the robots paths (must be run locally). Otherwise, robot paths are not calculated
parser = argparse.ArgumentParser(description="output robot paths.")

parser.add_argument(
    "--robot_path", 
    action="store_true", 
    help="Set OUTPUT_ROBOT_PATHS to True"
)

args = parser.parse_args()

OUTPUT_ROBOT_PATHS = args.robot_path

# Root repository
REPO_ROOT = Path(__file__).resolve().parent

# The path of the YOLOv8 model that is used to crop the scoreboard from the raw videos.
CROP_SCOREBOARD_MODEL_PATH = REPO_ROOT / "models" / "crop_scoreboard.pt" 

# The path of the YOLOv8 model that extracts the information from the cropped scoreboard.
SCOREBOARD_INFO_MODEL_PATH = REPO_ROOT / "models" / "extract_scoreboard_info.pt" 

# The path of the YOLOv8 model that identifies robots.
ROBOT_MODEL_PATH = REPO_ROOT / "models" / "best_tuned_yolov8.pt" 

# The path of the YOLOv8 model that identifies coral.
CORAL_MODEL_PATH = REPO_ROOT / "models" / "best_coral.pt" 

# The path of the YOLOv8 model that identifies the number on robots.
ROBOT_NUMBER_MODEL_PATH = REPO_ROOT / "models" / "best_number.pt" 

# Path to custom BoT-SORT tracker
CUSTOM_TRACKER_PATH = "botsort.yaml"

# How many frames are skipped between each processing step for the scoreboard. If the value is 15, it will process 1 frame every 15 frames.
SCOREBOARD_FRAME_SKIP = 15 

ROBOT_CLASS_ID = 1
BLUE_NUMBER_CLASS_ID = 0
RED_NUMBER_CLASS_ID = 1


# Whether to delete the downloaded video after processing to save space. Set to True to enable deletion.
DELETE_VIDEO = False 


# Initialize directories and models
os.makedirs("output", exist_ok=True)
os.makedirs("output/videos", exist_ok=True)

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
        "quiet": True,
        "no_warnings": True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return os.path.splitext(os.path.basename(filename))[0] + ".mp4"

# Download the video
video_filename = download_video(YOUTUBE_LINK, "output/videos")
video_path = os.path.join("output/videos", video_filename)

VIDEO_PATH = video_path

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
    crop_results = crop_model(frame, device=DEVICE, verbose = False)[0]

    # Skip if no scoreboard is found
    if len(crop_results.boxes) == 0:
        frame_idx += 1
        continue

    # Crop scoreboard from frame
    x1, y1, x2, y2 = map(int, crop_results.boxes.xyxy[0])
    scoreboard = frame[y1:y2, x1:x2]

    # Find the elements inside the cropped frame (scores, timer, team numbers)
    info_results = info_model(scoreboard, device=DEVICE, verbose = False)[0]

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

# Delete video after processing to save space if desired.
if DELETE_VIDEO == True and os.path.exists(video_path):
    os.remove(video_path)

# Convert to dataframe
scoreboard_df = pd.DataFrame(rows)

# VIDEO PROCESSING

# --- SETTINGS ---
input_folder = "output/videos"                # Folder containing your video
output_folder = "output/cropped_videos"       # Where the new video will go

# Method definitions

# Determines if the frame is part of the match (skipping intro screen).
# Detects edge density (low edge density = intro screen).
def is_real_match_frame(frame):
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corner_roi = gray[0:int(h*0.2), 0:int(w*0.2)]
    
    edges = cv2.Canny(corner_roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Debug print, used to adjust edge density threshold
    # print(f"Current Edge Density: {edge_density:.4f}")
    
    return edge_density > 0.04

# Determines if climb stage has started through the scoreboard. 
# If the timer is at 0:20 at the end of the video, climb has started.
def is_climb_stage(frame):
    climb = False
    
    crop_results = crop_model(frame, device = DEVICE, verbose = False)[0]
    if len(crop_results.boxes) == 0:
        return False
    
    x1, y1, x2, y2 = map(int, crop_results.boxes.xyxy[0])
    scoreboard = frame[y1:y2, x1:x2]

    info_results = info_model(scoreboard, device = DEVICE, verbose = False)[0]
    timer = None
    for b in info_results.boxes:

        cls_id = int(b.cls[0])
        label = info_model.names[cls_id]

        x1, y1, x2, y2 = map(int, b.xyxy[0])
        region = scoreboard[y1:y2, x1:x2]

        if label == "timer":
            timer = read_timer(region)

    if timer is not None:
        try:
            minutes = int(timer.split(":")[0])
            seconds = int(timer.split(":")[1])
            if minutes == 0 and seconds < 20:
                climb = True
        except:
            pass
    
    return climb


# Finds the y-coordinate split of the points of view and returns it.
# Uses Canny Edge Detection to capture dividing lines.
def find_split_by_edge_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return None

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    start_y, end_y = int(h * 0.45), int(h * 0.70)
    zone = gray[start_y:end_y, :]

    # Canny Edge Detection
    edges = cv2.Canny(zone, 50, 150)
    edge_sums = np.sum(edges, axis=1)
    relative_split_y = np.argmax(edge_sums)
    absolute_split_y = start_y + relative_split_y
    
    return int(absolute_split_y)

# Crops the video at the input path at the given y-coordinate.
# Saves the bottom portion to the output path.
def crop_and_save_video(input_path, output_path, split_y):
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    halfway = total_frames // 2
    frame_skip = 30
    
    new_height = int(orig_height - split_y)
    new_width = orig_width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    current = Path(output_path)
    if current.is_file():
        current.unlink()

    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    processing_started = False
    frame_count = 0
    end_processing = False

    # Video processing loop!
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            break

        # Start processing if intro screen has passed
        if not processing_started:
            if is_real_match_frame(frame):
                processing_started = True
            else:
                continue
        
        # Stop processing if climb stage has started
        if frame_count > halfway and frame_count % frame_skip == 0:
            if is_climb_stage(frame):
                end_processing = True
        
        if end_processing:
            break
        
        cropped_frame = frame[split_y:orig_height, 0:new_width]

        out.write(cropped_frame)
        
    cap.release()
    out.release()

# Single video processing
input_file = VIDEO_PATH
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = os.path.join(output_folder, f"cropped_{video_filename}")
split_y = find_split_by_edge_detection(input_file)
if split_y:
    crop_and_save_video(input_file, output_file, split_y)

CROPPED_VIDEO_PATH = output_file

# Homography code
if OUTPUT_ROBOT_PATHS:
    # Info taken from scoreboard
    CALIBRATION_FRAME_ID = scoreboard_df.loc[scoreboard_df['timer'] == '0:00', 'Frame'].iloc[0]
    RED_LOCATION = scoreboard_df["red_location"].mode()[0]
    RED_TEAMS_SCOREBOARD = [int(str(team).strip()) for team in scoreboard_df['red_team_numbers'].mode()[0]]
    BLUE_TEAMS_SCOREBOARD = [int(str(team).strip()) for team in scoreboard_df['blue_team_numbers'].mode()[0]]

    # Returns the color for a given team number.
    def get_team_color(team_name):
        colors = {
            RED_TEAMS_SCOREBOARD[0]: (30, 30, 255),   # Deep Red
            RED_TEAMS_SCOREBOARD[1]: (80, 80, 255),   # Lighter Red
            RED_TEAMS_SCOREBOARD[2]: (0, 100, 255),   # Orange
            BLUE_TEAMS_SCOREBOARD[0]: (255, 30, 30),   # Deep Blue
            BLUE_TEAMS_SCOREBOARD[1]: (255, 150, 0),   # Cyan
            BLUE_TEAMS_SCOREBOARD[2]: (255, 0, 150),   # Purple
        }
        return colors.get(team_name, (128, 128, 128))

    RED_UI_COLOR = (30, 30, 255)
    BLUE_UI_COLOR = (255, 100, 30)

    # dimensions for the points that are clicked
    world_corners_local = np.array([
        [-3.15094, 13.05740], 
        [-1.57547, 10.32860], 
        [ 1.57547, 10.32860], 
        [ 3.15094, 13.05740]  
    ], dtype=np.float32)

    # Variables for calibration UI interaction
    clicked_points = []
    zoom = 1.0
    min_zoom = 1.0
    max_zoom = 6.0
    center_x = 0
    center_y = 0
    W_img = 0
    H_img = 0
    calibration_frame = None
    calibration_done = False 
    map_scale = 15
    map_width, map_height = 1200, 600
    trajectories = defaultdict(list)
    DYNAMIC_ID_TO_TEAM = {}
    id_votes = defaultdict(lambda: defaultdict(int))

    # Return zoomed view of calibration frame
    def get_view():
        global zoom, center_x, center_y, W_img, H_img, calibration_frame
        win_w = int(W_img / zoom)
        win_h = int(H_img / zoom)
        x1 = max(0, min(W_img - win_w, center_x - win_w // 2))
        y1 = max(0, min(H_img - win_h, center_y - win_h // 2))
        crop = calibration_frame[y1:y1+win_h, x1:x1+win_w]
        view = cv2.resize(crop, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
        return view, x1, y1

    # Mouse callback for selecting calibration points and zooming
    def mouse(event, x, y, flags, param):
        global clicked_points, zoom, center_x, center_y, W_img, H_img
        view, x1, y1 = get_view()

        def to_img(px, py):
            return (x1 + px / W_img * (W_img / zoom), y1 + py / H_img * (H_img / zoom))

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 8: 
                ix, iy = to_img(x, y)
                clicked_points.append([ix, iy])
                side = "LEFT" if len(clicked_points) <= 4 else "RIGHT"
                num = len(clicked_points) if len(clicked_points) <= 4 else len(clicked_points) - 4
                print(f"Recorded {side} reference point {num}: ({ix:.2f}, {iy:.2f})")
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            if clicked_points:
                print("Removed:", clicked_points.pop())
        elif event == cv2.EVENT_MOUSEWHEEL:
            ix, iy = to_img(x, y)
            if flags > 0:
                zoom = min(max_zoom, zoom * 1.2)
            else:
                zoom = max(min_zoom, zoom / 1.2)
            center_x, center_y = int(ix), int(iy)

    # Initialize video capture and jump to calibration frame
    cap = cv2.VideoCapture(CROPPED_VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {CROPPED_VIDEO_PATH}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME_ID)
    ret, calibration_frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {CALIBRATION_FRAME_ID}.")

    # Extract frame dimensions and initialize view center
    H_img, W_img = calibration_frame.shape[:2]
    center_x, center_y = W_img // 2, H_img // 2
    split_line_x = W_img / 2.0 
    esc_press_count = 0

    # Setup calibration window and mouse interaction
    cv2.namedWindow("Calibration Phase", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration Phase", mouse)

    print("\n--- DUAL CAMERA CALIBRATION ---")
    print("1. Select your 4 reference points on the LEFT field segment.")
    print("2. Select the SAME 4 reference points on the RIGHT field segment.")
    print("3. Press [ENTER] when all 8 points have been designated.")

    # Wait for user to select 8 calibration points and setup UI for point selection
    while not calibration_done:
        view, x1, y1 = get_view()
        
        split_view_x = int((split_line_x - x1) / (W_img / zoom) * W_img)
        cv2.line(view, (split_view_x, 0), (split_view_x, H_img), (255, 255, 255), 1, cv2.LINE_AA)

        for i, (px, py) in enumerate(clicked_points):
            sx = int((px - x1) / (W_img / zoom) * W_img)
            sy = int((py - y1) / (H_img / zoom) * H_img)
            
            is_left_side = i < 4
            if red_location.lower() == 'left':
                color = RED_UI_COLOR if is_left_side else BLUE_UI_COLOR
            else:
                color = BLUE_UI_COLOR if is_left_side else RED_UI_COLOR
            
            ch_size = 12
            cv2.line(view, (sx - ch_size, sy), (sx + ch_size, sy), color, 2, cv2.LINE_AA)
            cv2.line(view, (sx, sy - ch_size), (sx, sy + ch_size), color, 2, cv2.LINE_AA)
            cv2.circle(view, (sx, sy), 2, (255, 255, 255), -1, cv2.LINE_AA) # Center dot
            
            label = str(i + 1) if i < 4 else str(i - 3)
            cv2.putText(view, label, (sx + 8, sy - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        ui_height = 90
        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (W_img, ui_height), (15, 15, 15), -1)
        cv2.line(overlay, (0, ui_height), (W_img, ui_height), (0, 140, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, view, 0.15, 0, view)

        if len(clicked_points) < 4:
            title_text = "PHASE 1: LEFT FIELD CALIBRATION"
            sub_text = f"Select point {len(clicked_points) + 1} of 4 on the LEFT camera feed."
            status_color = RED_UI_COLOR if red_location.lower() == 'left' else BLUE_UI_COLOR
        elif len(clicked_points) < 8:
            title_text = "PHASE 2: RIGHT FIELD CALIBRATION"
            sub_text = f"Select point {len(clicked_points) - 3} of 4 on the RIGHT camera feed."
            status_color = BLUE_UI_COLOR if red_location.lower() == 'left' else RED_UI_COLOR
        else:
            title_text = "CALIBRATION COMPLETE"
            sub_text = "Press [ENTER] on your keyboard to finalize and begin tracking."
            status_color = (255, 255, 255)

        cv2.putText(view, title_text, (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 1, cv2.LINE_AA)
        cv2.putText(view, sub_text, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Calibration Phase", view)
        
        # Keyboard listener
        key = cv2.waitKey(1) & 0xFF
        if key in [10, 13] and len(clicked_points) == 8:
            calibration_done = True
        elif key == 27:
            esc_press_count += 1

            if esc_press_count <= 2:
                new_frame_id = CALIBRATION_FRAME_ID + 25 * esc_press_count
                print(f"Switching to calibration frame {new_frame_id}")

                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_id)
                ret, calibration_frame = cap.read()

                if not ret:
                    print(f"Could not read frame {new_frame_id}. Exiting.")
                    break

                clicked_points.clear()
                center_x, center_y = W_img // 2, H_img // 2

            else:
                print("Calibration cancelled.")
                break

    cv2.destroyAllWindows()

    if len(clicked_points) != 8:
        print("Calibration not completed. Skipping homography.")
        cap.release()
        cv2.destroyAllWindows()
    else:
        pts_left = np.array(clicked_points[:4], dtype=np.float32)
        pts_right = np.array(clicked_points[4:], dtype=np.float32)

        H_left, _ = cv2.findHomography(pts_left, world_corners_local)
        H_right, _ = cv2.findHomography(pts_right, world_corners_local)

        print("\nLeft and Right Homography matrices computed successfully.\n")

    # Function to draw the field overlay on the top-down map
    def draw_field_overlay(canvas, scale, width, height):
        line_color = (255, 255, 255) 
        thickness = 2

        # Helper function to convert real-world feet coordinates to pixel coordinates on the canvas
        def to_px(x_ft, y_ft):
            px_x = int(x_ft * scale) + (width // 2)
            px_y = int(y_ft * scale) + (height // 2) 
            return (px_x, px_y)

        wall_x_right = 0.95 + 27.7297
        wall_y_max = 18.1037 / 2.0
        
        dx = 7.0685 * np.sin(np.radians(54))
        dy = 7.0685 * np.cos(np.radians(54))
        
        corner_x_right = wall_x_right - dx
        corner_y_max = wall_y_max + dy
        
        wall_x_left = -wall_x_right
        corner_x_left = -corner_x_right
        
        perimeter_pts_ft = [
            (0, corner_y_max),                 
            (corner_x_right, corner_y_max),    
            (wall_x_right, wall_y_max),        
            (wall_x_right, -wall_y_max),       
            (corner_x_right, -corner_y_max),   
            (corner_x_left, -corner_y_max),    
            (wall_x_left, -wall_y_max),        
            (wall_x_left, wall_y_max),         
            (corner_x_left, corner_y_max),     
        ]
        
        perimeter_px = np.array([to_px(x, y) for x, y in perimeter_pts_ft], np.int32)
        cv2.polylines(canvas, [perimeter_px], isClosed=True, color=line_color, thickness=thickness)

        hex_side = 3.15094154
        hex_dx = hex_side * (np.sqrt(3) / 2.0)
        hex_dy = hex_side / 2.0
        
        right_hex_center_x = 0.95 + 13.0574
        left_hex_center_x = -right_hex_center_x
        
        for cx in [left_hex_center_x, right_hex_center_x]:
            hex_pts_ft = [
                (cx, hex_side),             
                (cx + hex_dx, hex_dy),      
                (cx + hex_dx, -hex_dy),     
                (cx, -hex_side),            
                (cx - hex_dx, -hex_dy),     
                (cx - hex_dx, hex_dy)       
            ]
            hex_px = np.array([to_px(x, y) for x, y in hex_pts_ft], np.int32)
            cv2.polylines(canvas, [hex_px], isClosed=True, color=line_color, thickness=thickness)

        cv2.line(canvas, to_px(0, corner_y_max), to_px(0, -corner_y_max), (100, 100, 100), 1, cv2.LINE_AA)

# Load models
robot_model = YOLO(ROBOT_MODEL_PATH)
number_model = YOLO(ROBOT_NUMBER_MODEL_PATH)

# BoT-SORT
def run_botsort(model_path, video_path, tracker_path, save_video=False, output_dir=None, stride=1):

    model = YOLO(model_path)

    print("Running BotSort tracking...")

    results = model.track(
        source=video_path,
        tracker=tracker_path,
        stream=True,
        persist=True,
        conf=0.35,
        device=0,
        vid_stride=stride,
        verbose=False
    )

    tracking_results = []
    video_writer = None

    for frame_i, result in enumerate(results):

        if result.boxes is None or result.boxes.xyxy is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        ids = result.boxes.id

        if ids is None:
            continue

        ids = ids.cpu().numpy().astype(int)

        robot_indices = [i for i, c in enumerate(classes) if c == ROBOT_CLASS_ID]

        frame = result.plot()

        for i in robot_indices:

            x1, y1, x2, y2 = boxes[i]
            track_id = ids[i]

            # Store result
            tracking_results.append({
                "frame": frame_i * stride,
                "track_id": int(track_id),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "crop": frame[int(y1):int(y2), int(x1):int(x2)].copy()
            })

            # Draw
            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if save_video:
            if video_writer is None:
                h, w = frame.shape[:2]
                output_path = output_dir / "botsort_with_ids.mp4"

                video_writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (w, h)
                )

            video_writer.write(frame)

    if video_writer:
        video_writer.release()

    print("Tracking complete.")
    return tracking_results

# Image processing functions
def estimate_angle_from_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]

    if angle < -45:
        angle += 90

    return angle


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_crop(crop, threshold):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    angle = estimate_angle_from_crop(crop)
    rotated = rotate_image(thresh, angle)

    return rotated

def preprocess_variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    variants = []

    # Simple
    for t in [150, 165, 180, 195, 210]:
        _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        variants.append(th)

    # Adaptive
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    variants.append(adaptive)

    # Otsu
    _, otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(otsu)

    # 4. Histogram
    eq = cv2.equalizeHist(gray)
    _, th_eq = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(th_eq)

    return variants

def find_closest_frame(target_frame, available_frames):
    return min(available_frames, key=lambda f: abs(f - target_frame))

def assign_coral_to_robot(coral_df, tracking_lookup, final_labels):

    assigned_rows = []

    available_frames = sorted(tracking_lookup.keys())

    for _, row in coral_df.iterrows():

        coral_frame = row["frame"]
        coral_x = row["x"]
        coral_y = row["y"]

        if coral_x is None or coral_y is None:
            assigned_rows.append({**row, "robot_id": None})
            continue

        # Closest frame
        closest_frame = find_closest_frame(coral_frame, available_frames)

        tracks = tracking_lookup.get(closest_frame, [])

        best_dist = float("inf")
        best_robot = None
        best_track_id = None

        # Closest robot
        for t in tracks:
            x1, y1, x2, y2 = t["box"]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            dx = cx - coral_x
            dy = cy - coral_y

            dist = math.sqrt((dx**2) * 2.0 + (dy**2) * 0.5)

            if dist < best_dist:
                best_dist = dist
                best_track_id = t["track_id"]

        # Map to number
        if best_track_id in final_labels:
            best_robot = final_labels[best_track_id]
        else:
            best_robot = None

        assigned_rows.append({
            **row,
            "closest_frame": closest_frame,
            "track_id": best_track_id,
            "robot_id": best_robot,
            "distance": best_dist
        })

    return pd.DataFrame(assigned_rows)

# OCR
def read_number_from_image(img):
    variants = preprocess_variants(img)
    guesses = []

    angle = estimate_angle_from_crop(img)
    for processed in variants:
        rotated = rotate_image(processed, angle)

        results = reader.readtext(
            rotated,
            allowlist="0123456789",
            detail=1,
            paragraph=False
        )

        results = [r for r in results if r[2] > 0.5]

        if results:
            guesses.append(results[0][1])

    if not guesses:
        return None

    c = Counter(guesses)
    max_count = max(c.values())

    tied = [num for num, count in c.items() if count == max_count]

    return max(tied, key=lambda x: len(str(x)))

# Matching system
def levenshtein(a, b):
    a, b = str(a), str(b)
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[-1][-1]


def similarity(a, b):
    if not a or not b:
        return 0
    dist = levenshtein(a, b)
    return 1 - dist / max(len(str(a)), len(str(b)))


def alignment_score(a, b):
    a, b = str(a), str(b)
    best = 0

    for shift in range(-len(b), len(a)+1):
        matches = 0
        for i in range(len(a)):
            j = i - shift
            if 0 <= j < len(b) and a[i] == b[j]:
                matches += 1
        best = max(best, matches)

    return best / max(len(a), len(b))


def combined_score(a, b, w1=0.7, w2=0.3):
    return w1 * similarity(a, b) + w2 * alignment_score(a, b)


def match_number_single(detected, nums):
    if detected is None:
        return None

    best_score = 0.5
    match = None

    for n in nums:
        current = combined_score(detected, n)
        if current > best_score:
            best_score = current
            match = n

    return match

def apply_elimination(final_labels, track_candidates, all_ids, tracks):
    assigned_ids = set(final_labels.values())
    remaining_ids = set(all_ids) - assigned_ids

    unlabeled_tracks = [t for t in tracks if t not in final_labels]

    # Perfect elimination
    if len(unlabeled_tracks) == len(remaining_ids):
        for t, rid in zip(unlabeled_tracks, remaining_ids):
            final_labels[t] = rid

    # Candidate filtering
    else:
        for t in unlabeled_tracks:
            candidates = set(track_candidates.get(t, []))
            candidates -= assigned_ids

            if len(candidates) == 1:
                final_labels[t] = candidates.pop()

    return final_labels

# Number reading
def read_numbers(tracking_results, blue_team_numbers=[], red_team_numbers=[], stride=1):

    results = []
    current_frame = None
    frame_data = []

    for entry in tracking_results:

        frame_idx = entry["frame"]
        robot_crop = entry["crop"]

        if frame_idx % stride != 0:
            continue

        if current_frame is None:
            current_frame = frame_idx

        if frame_idx != current_frame:
            # print(f"Frame {current_frame}: {frame_data}")
            results.append({
                "frame": current_frame,
                "detections": frame_data
            })
            frame_data = []
            current_frame = frame_idx

        if robot_crop is None or robot_crop.size == 0:
            continue

        number_results = number_model(robot_crop, verbose=False)[0]

        for nbox in number_results.boxes:
            cls_id = int(nbox.cls[0])
            if cls_id == RED_NUMBER_CLASS_ID:
                team_list = red_team_numbers
            elif cls_id == BLUE_NUMBER_CLASS_ID:
                team_list = blue_team_numbers
            else:
                continue

            if nbox.conf[0] < 0.6:
                continue

            nx1, ny1, nx2, ny2 = map(int, nbox.xyxy[0])


            pad = 5
            h2, w2 = robot_crop.shape[:2]
            nx1 = max(0, nx1 - pad)
            ny1 = max(0, ny1 - pad)
            nx2 = min(w2, nx2 + pad)
            ny2 = min(h2, ny2 + pad)

            if (nx2 - nx1) < 30 or (ny2 - ny1) < 15:
                continue

            number_crop = robot_crop[ny1:ny2, nx1:nx2]

            if number_crop.size == 0:
                continue

            detected = read_number_from_image(number_crop)
            matched = match_number_single(detected, team_list)

            # homography
            if OUTPUT_ROBOT_PATHS:
                if matched is not None:
                    id_votes[entry["track_id"]][matched] += 1

            frame_data.append({
                "detected": detected,
                "matched": matched,
                "track_id": entry["track_id"],
                "box": entry["box"],
                "alliance_hint": "blue" if cls_id == BLUE_NUMBER_CLASS_ID else "red"
            })

    if frame_data:
        # print(f"Frame {current_frame}: {frame_data}")
        results.append({
            "frame": current_frame,
            "detections": frame_data
        })

    return results

# Identification and Scoring Pipeline
def run_full_pipeline(
    video_path,
    robot_model_path,
    number_model_path,
    tracker_path,
    blue_ids,
    red_ids,
    output_path,
    tracking_stride=1,
    ocr_stride=1
):
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    print("Running BotSort...")
    botsort_results = run_botsort(
        robot_model_path,
        video_path,
        tracker_path,
        stride=tracking_stride
    )

    print("Reading numbers...")
    output = read_numbers(
        botsort_results,
        blue_ids,
        red_ids,
        stride=ocr_stride
    )

    # Alliance detection
    track_alliance_votes = defaultdict(lambda: {"blue": 0, "red": 0})

    for frame in output:
        for det in frame["detections"]:
            tid = det["track_id"]
            match = det["matched"]

            hint = det.get("alliance_hint")
            if hint == "blue":
                track_alliance_votes[tid]["blue"] += 2
            elif hint == "red":
                track_alliance_votes[tid]["red"] += 2

            if match in blue_ids:
                track_alliance_votes[tid]["blue"] += 1
            elif match in red_ids:
                track_alliance_votes[tid]["red"] += 1

    track_alliance = {}

    for tid, votes in track_alliance_votes.items():
        if votes["blue"] > votes["red"]:
            track_alliance[tid] = "blue"
        elif votes["red"] > votes["blue"]:
            track_alliance[tid] = "red"
        else:
            track_alliance[tid] = None

    # Track memory
    track_memory = defaultdict(list)

    for frame in output:
        for det in frame["detections"]:
            tid = det["track_id"]
            match = det["matched"]

            if match is None:
                continue

            alliance = track_alliance.get(tid)

            if alliance == "blue" and match in blue_ids:
                track_memory[tid].append(match)
            elif alliance == "red" and match in red_ids:
                track_memory[tid].append(match)

    # 3: Global unique matching
    all_tracks = list(track_memory.keys())
    all_ids = list(blue_ids) + list(red_ids)

    if len(all_tracks) == 0:
        final_labels = {}
    else:
        cost_matrix = np.full((len(all_tracks), len(all_ids)), fill_value=1e6)

        for i, tid in enumerate(all_tracks):
            nums = track_memory[tid]
            if not nums:
                continue

            scores = defaultdict(float)

            # Recency Weighting
            for j, num in enumerate(nums):
                weight = 0.9 ** (len(nums) - j)
                scores[num] += weight

            total_weight = sum(scores.values())
            if total_weight == 0:
                continue

            alliance = track_alliance.get(tid)

            for j, rid in enumerate(all_ids):

                if alliance == "blue" and rid not in blue_ids:
                    continue
                if alliance == "red" and rid not in red_ids:
                    continue

                if rid in scores:
                    confidence = scores[rid] / total_weight

                    if confidence < 0.4:
                        continue

                    cost_matrix[i, j] = -confidence  # maximize

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        final_labels = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0:
                final_labels[all_tracks[r]] = all_ids[c]

    print("Final track labels:", final_labels)

    # Build lookup
    tracking_lookup = defaultdict(list)

    for entry in botsort_results:
        tracking_lookup[entry["frame"]].append(entry)

    # Draw video
    print("Drawing Video...")

    cap = cv2.VideoCapture(video_path)
    video_writer = None

    frame_idx = 0
    last_drawn = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx in tracking_lookup:
            draw_frame = frame.copy()

            for entry in tracking_lookup[frame_idx]:
                tid = entry["track_id"]
                x1, y1, x2, y2 = map(int, entry["box"])

                # homography
                if OUTPUT_ROBOT_PATHS:

                    if id_votes[tid]:
                        assigned_team = max(id_votes[tid], key=id_votes[tid].get)
                        DYNAMIC_ID_TO_TEAM[tid] = assigned_team

                    bc_x = (x1+x2) / 2
                    bc_y = y2
                    pts = np.array([[[bc_x, bc_y]]], dtype=np.float32)
                    if bc_x < split_line_x:
                        transformed = cv2.perspectiveTransform(pts, H_left)
                        rx, ry = transformed[0][0]
                        rot_x, rot_y = -ry, -rx
                        real_x_ft, real_y_ft = rot_x - 0.95, rot_y
                    else:
                        transformed = cv2.perspectiveTransform(pts, H_right)
                        rx, ry = transformed[0][0]
                        rot_x, rot_y = ry, rx
                        real_x_ft, real_y_ft = rot_x + 0.95, rot_y
                    map_x = int(real_x_ft * map_scale) + (map_width // 2)
                    map_y = int(real_y_ft * map_scale) + (map_height // 2)
                    if 0 <= map_x < map_width and 0 <= map_y < map_height:
                        trajectories[tid].append((map_x, map_y))

                alliance = track_alliance.get(tid)

                if alliance == "blue":
                    color = (255, 0, 0)
                elif alliance == "red":
                    color = (0, 0, 255)
                else:
                    color = (200, 200, 200)

                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)

                if tid in final_labels:
                    text = str(final_labels[tid])
                else:
                    text = f"ID {tid}"

                cv2.putText(
                    draw_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            last_drawn = draw_frame

        if last_drawn is None:
            continue

        if video_writer is None:
            h, w = last_drawn.shape[:2]
            video_writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (w, h)
            )

        video_writer.write(last_drawn)

    cap.release()

    if video_writer:
        video_writer.release()

    print("Pipeline complete. Video saved to:", output_path)

    return {
        "tracking": botsort_results,
        "ocr_output": output,
        "final_labels": final_labels
    }

# Coral
VIDEO_FOLDER = 'output/videos'

SLOT_CAPACITIES = [3, 1, 2, 2, 1, 3]
NUM_SLOTS = len(SLOT_CAPACITIES)
NUM_REEFS = 2

HISTORY_LEN = 5
MIN_CONF = 0.5
TOP_ZONE_RATIO = 0.20

CONFIRM_FRAMES = 3

EMPTY = 0
FILLING = 1
FILLED = 2

coral_model = YOLO(CORAL_MODEL_PATH)

# Coral: Helpers
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2)/2, (y1 + y2)/2)

def assign_to_reef_and_slot(cx, cy, reefs):
    for r_idx, r in enumerate(reefs):
        rx1, ry1, rx2, ry2 = r
        rw, rh = rx2 - rx1, ry2 - ry1

        if (rx1 < cx < rx2) and (ry1 < cy < ry1 + rh*TOP_ZONE_RATIO):
            rel_x = cx - rx1
            slot_width = rw / NUM_SLOTS
            slot_margin = slot_width * 0.1
            slot = int((rel_x - slot_margin)/slot_width)
            slot = max(0, min(slot, NUM_SLOTS-1))
            return r_idx, slot
    return None, None

# Coral: Main
video_file = os.path.basename(CROPPED_VIDEO_PATH)

print(f"\nProcessing: {video_file}")

cap = cv2.VideoCapture(CROPPED_VIDEO_PATH)
print(f"\nProcessing: {video_file}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cutoff_frame = total_frames - int(49*fps)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Coral: State
reef_grids = [[0]*NUM_SLOTS for _ in range(NUM_REEFS)]
history = [[deque(maxlen=HISTORY_LEN) for _ in range(NUM_SLOTS)] for _ in range(NUM_REEFS)]

states = [[EMPTY]*NUM_SLOTS for _ in range(NUM_REEFS)]
timers = [[0]*NUM_SLOTS for _ in range(NUM_REEFS)]

total_points = 0
frame_idx = 0

events = []

# Coral: Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= cutoff_frame:
        break

    cropped = frame
    results = coral_model(cropped, conf=MIN_CONF, verbose=False)

    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    clss  = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes else []
    confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []

    reefs = [boxes[i] for i, c in enumerate(clss) if coral_model.names[c]=='reef']
    reefs.sort(key=lambda x: x[0])

    if len(reefs) < NUM_REEFS:
        frame_idx += 1
        continue

    reefs = reefs[:NUM_REEFS]

    # Count raw
    raw_counts = [[0]*NUM_SLOTS for _ in range(NUM_REEFS)]
    raw_positions = [[[] for _ in range(NUM_SLOTS)] for _ in range(NUM_REEFS)]

    for i, c in enumerate(clss):
        if coral_model.names[c] != 'coral' or confs[i] < MIN_CONF:
            continue

        cx, cy = get_center(boxes[i])
        r_idx, slot = assign_to_reef_and_slot(cx, cy, reefs)

        if (
            r_idx is not None and
            slot is not None and
            0 <= r_idx < NUM_REEFS and
            0 <= slot < NUM_SLOTS
        ):
            raw_counts[r_idx][slot] += 1
            raw_positions[r_idx][slot].append((cx, cy))

    # Smooth
    smoothed = [[0]*NUM_SLOTS for _ in range(NUM_REEFS)]
    for r in range(NUM_REEFS):
        for s in range(NUM_SLOTS):
            history[r][s].append(raw_counts[r][s])
            smoothed[r][s] = int(round(np.median(history[r][s])))

    # Markov state update
    DECAY = 1
    GROWTH = 1
    THRESHOLD = 4

    for r in range(NUM_REEFS):
        for s in range(NUM_SLOTS):

            curr = smoothed[r][s]
            placed = reef_grids[r][s]

            if placed >= SLOT_CAPACITIES[s]:
                continue

            stable = history[r][s].count(curr) >= 3

            if curr >= placed + 1 and stable:
                timers[r][s] += GROWTH
            else:
                timers[r][s] -= DECAY

            timers[r][s] = max(-THRESHOLD, timers[r][s])

            # Event detection
            if timers[r][s] >= THRESHOLD:
                reef_grids[r][s] += 1
                total_points += 4

                # Estimate position
                if raw_positions[r][s]:
                    avg_x = int(np.mean([p[0] for p in raw_positions[r][s]]))
                    avg_y = int(np.mean([p[1] for p in raw_positions[r][s]]))
                else:
                    avg_x, avg_y = None, None

                # Log event
                events.append({
                    "frame": frame_idx,
                    "reef": r,
                    "slot": s,
                    "new_count": reef_grids[r][s],
                    "x": avg_x,
                    "y": avg_y
                })

                timers[r][s] = -THRESHOLD

    frame_idx += 1

cap.release()

# Coral: Results
print(f"Total points: {total_points}")
for r in range(NUM_REEFS):
    print(f"Reef {r}: {reef_grids[r]}")

coral_df = pd.DataFrame(events)

print("\nEvent DataFrame:")
print(coral_df)

# Save CSV
output_csv = Path("output") / f"{video_file}_coral_events.csv"
coral_df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

# Get team numbers
counts = Counter(team for teams in scoreboard_df["blue_team_numbers"] for team in teams)
majority = counts.most_common(3)
BLUE_IDS = [majority[0][0], majority[1][0], majority[2][0]]
counts = Counter(team for teams in scoreboard_df["red_team_numbers"] for team in teams)
majority = counts.most_common(3)
RED_IDS = [majority[0][0], majority[1][0], majority[2][0]]

# RUN
scoring_dir = Path("output/scoring")
scoring_dir.mkdir(parents=True, exist_ok=True)
output_dir = Path("output/tracking")
output_dir.mkdir(parents=True, exist_ok=True)


botsort_results = run_full_pipeline(
    CROPPED_VIDEO_PATH,
    ROBOT_MODEL_PATH,
    ROBOT_NUMBER_MODEL_PATH,
    CUSTOM_TRACKER_PATH,
    BLUE_IDS,
    RED_IDS,
    output_path=output_dir / "final_output.mp4",
    tracking_stride=3,
    ocr_stride=15
)


tracking_lookup = defaultdict(list)


for entry in botsort_results["tracking"]:
    tracking_lookup[entry["frame"]].append(entry)




assigned_df = assign_coral_to_robot(
    coral_df,
    tracking_lookup,
    botsort_results["final_labels"]
)


print(assigned_df)


unknown_events = assigned_df[assigned_df["robot_id"].isna()].copy()




def review_all_events(
    video_path,
    assigned_df,
    tracking_lookup,
    final_labels,
    blue_ids,
    red_ids
):
    cap = cv2.VideoCapture(video_path)


    all_robot_options = list(blue_ids) + list(red_ids)
    user_labels = []


    for idx, row in assigned_df.iterrows():


        frame_num = int(row["closest_frame"])


        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue


        draw = frame.copy()


        tracks = tracking_lookup.get(frame_num, [])


        # Find closest robot (scorer)
        best_tid = None
        best_dist = float("inf")


        if row["x"] is not None and row["y"] is not None:
            coral_x, coral_y = row["x"], row["y"]


            for t in tracks:
                x1, y1, x2, y2 = t["box"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2


                # Weighted distance (favor X)
                dx = abs(cx - coral_x)
                dy = abs(cy - coral_y)
                dist = dx * 2.5 + dy * 0.5


                if dist < best_dist:
                    best_dist = dist
                    best_tid = t["track_id"]


        # Draw robots
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["box"])
            tid = t["track_id"]


            if tid in final_labels:
                label = final_labels[tid]
            else:
                label = f"ID {tid}"


            # Color priority
            if tid == best_tid:
                color = (0, 255, 0)
            elif tid in final_labels:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)


            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, str(label), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        # Draw coral
        if row["x"] is not None and row["y"] is not None:
            cx, cy = int(row["x"]), int(row["y"])


            if pd.isna(row["robot_id"]):
                coral_color = (255, 0, 0)
            else:
                coral_color = (0, 255, 0)


            cv2.circle(draw, (cx, cy), 8, coral_color, -1)


        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


        # Show frame
        filename = scoring_dir / (
            f"event_{idx}_frame_{frame_num}_"
            f"curr_{row['robot_id']}_closest_{best_tid}.png"
        )
        cv2.imwrite(str(filename), draw)
        print(f"File Name: {filename}")


        # User input (if unknown)
        if pd.isna(row["robot_id"]):


            print("Select robot:")
            for i, rid in enumerate(all_robot_options):
                print(f"{i+1}: {rid}")
            print("0: Unknown")


            choice = input("Enter choice: ")


            if choice == "0":
                chosen = None
            elif choice.isdigit() and 1 <= int(choice) <= len(all_robot_options):
                chosen = all_robot_options[int(choice) - 1]
            else:
                print("Invalid → keeping Unknown")
                chosen = None


            user_labels.append({
                "index": idx,
                "robot_id": chosen
            })


    cap.release()
    return user_labels


user_labels = review_all_events(
    CROPPED_VIDEO_PATH,
    assigned_df,
    tracking_lookup,
    botsort_results["final_labels"],
    BLUE_IDS,
    RED_IDS
)


for entry in user_labels:
    idx = entry["index"]
    assigned_df.loc[idx, "robot_id"] = entry["robot_id"]


scoring_counts = (
    assigned_df
    .dropna(subset=["robot_id"])
    .groupby("robot_id")
    .size()
    .reset_index(name="coral_scored")
)

# Save CSV
output_csv = Path("output") / f"{video_file}_scores.csv"
scoring_counts.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

print(scoring_counts)

# Draw the homography map and place trajectories. Trajectories will be layered so that known team number's trajectories will be placed on top. Also there's a static legend for the 6 known team numbers.
if OUTPUT_ROBOT_PATHS:
    final_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    draw_field_overlay(final_map, map_scale, map_width, map_height)

    mapped_segments = defaultdict(list)
    unmapped_segments = []

    for track_id, points in trajectories.items():
        if len(points) < 2: 
            continue 
        
        team = DYNAMIC_ID_TO_TEAM.get(track_id)
        if team in RED_TEAMS_SCOREBOARD + BLUE_TEAMS_SCOREBOARD:
            mapped_segments[team].append(points)
        else:
            unmapped_segments.append(points)

    overlay = final_map.copy()
    unmapped_color = (150, 150, 150)

    for points in unmapped_segments:
        pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts_array], isClosed=False, color=unmapped_color, thickness=1, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.4, final_map, 0.6, 0, final_map)

    for team, segments in mapped_segments.items():
        color = get_team_color(team)
        for points in segments:
            pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(final_map, [pts_array], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    legend_x = 20
    legend_y = 30

    for team in RED_TEAMS_SCOREBOARD + BLUE_TEAMS_SCOREBOARD:
        color = get_team_color(team)
        cv2.line(final_map, (legend_x, legend_y - 5), (legend_x + 25, legend_y - 5), color, 3)
        cv2.putText(final_map, f"Team: {team}", (legend_x + 35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        legend_y += 30
        if legend_y > map_height - 30:
            legend_y = 30
            legend_x += 140

    top_match_line_y_px = int(-(18.1037 / 2.0 + 7.0685 * np.cos(np.radians(54))) * map_scale) + (map_height // 2)
    label_y = top_match_line_y_px - 25

    left_label_x = (map_width // 4) - 80 
    right_label_x = (3 * map_width // 4) - 80

    if red_location.lower() == 'left':
        cv2.putText(final_map, "RED ALLIANCE", (left_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, RED_UI_COLOR, 2)
        cv2.putText(final_map, "BLUE ALLIANCE", (right_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, BLUE_UI_COLOR, 2)
    else:
        cv2.putText(final_map, "BLUE ALLIANCE", (left_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, BLUE_UI_COLOR, 2)
        cv2.putText(final_map, "RED ALLIANCE", (right_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, RED_UI_COLOR, 2)

    base_name, ext = os.path.splitext("robot_trajectories_map.png")
    counter = 1
    final_output_name = "robot_trajectories_map.png"
    while os.path.exists(final_output_name):
        final_output_name = f"{base_name} ({counter}){ext}"
        counter += 1

    cv2.imshow("Final Trajectories - Press any key to close", final_map)
    cv2.imwrite(final_output_name, final_map) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print(f"Final map saved as '{final_output_name}'. Process finished.")
