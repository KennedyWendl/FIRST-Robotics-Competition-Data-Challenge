# The user will click 8 points, 4 on the left reef and then 4 on the right reef. They will click the 4 corners of the reef that are touching the flat field
# From a top-down view you'd click as follows:
#
#    LEFT REEF         RIGHT REEF   
#
#  1             4    5             8
#   \           /      \           / 
#    \         /        \         /
#     2-------3          6-------7
#

import cv2
import numpy as np
import os
import easyocr
from collections import defaultdict, Counter
from ultralytics import YOLO

# Paths to the video and name of robot trajectory image
video_path = "videos/cropped_Final 1 - 2025 Central Missouri Regional.mp4"
image_output_name = "robot_trajectories_map.png"

# Define alliance team numbers and location of the alliances. calibration_frame_idx sets what frame to select for selecting points. Will also be automated using scoreboard
# Will be integrated with the scoreboard to make it automatic.
RED_LOCATION = 'right' 
RED_TEAMS = [4522, 5968, 1756]
BLUE_TEAMS = [5801, 6419, 3928]
calibration_frame_idx = 200

# Frequency of performing OCR on tracked robots for mapping team numbers. Reads every X frames
OCR_INTERVAL = 30

# Used for testing. Set to False to disable the video feed, and True shows it.
SHOW_VIDEO_FEED = True

# Models paths
ROBOT_MODEL_PATH = "models/best_tuned_yolov8.pt"
NUMBER_MODEL_PATH = "models/best_number.pt"
tracker_config = "models/botsort_custom.yaml"

# Classes used by models
ROBOT_CLASS_IDS = [1]
BLUE_NUMBER_CLASS_ID = 0
RED_NUMBER_CLASS_ID = 1

# Load models
print("Loading Models...")
robot_model = YOLO(ROBOT_MODEL_PATH)
number_model = YOLO(NUMBER_MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True) 

# Combine all team numbers to one list
ALL_TEAMS = RED_TEAMS + BLUE_TEAMS

# data structures for mapping ID to the team numbers
id_votes = defaultdict(lambda: defaultdict(int))
DYNAMIC_ID_TO_TEAM = {}

# Returns the color for a given team number.
def get_team_color(team_name):
    colors = {
        4522: (30, 30, 255),   # Deep Red
        5968: (80, 80, 255),   # Lighter Red
        1756: (0, 100, 255),   # Orange
        5801: (255, 30, 30),   # Deep Blue
        6419: (255, 150, 0),   # Cyan
        3928: (255, 0, 150),   # Purple
    }
    return colors.get(team_name, (128, 128, 128))

# Define UI Colors (BGR format) for Calibration and Labels
RED_UI_COLOR = (30, 30, 255)
BLUE_UI_COLOR = (255, 100, 30)

# Estimate rotation angle of number region for OCR improvement
def estimate_angle_from_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
        
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    
    if angle < -45:
        angle += 90
    return angle

# Rotate image by a given angle
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Generate multiple thresholded versions of image for OCR robustness
def preprocess_variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants = []

    for t in [150, 165, 180, 195, 210]:
        _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        variants.append(th)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    eq = cv2.equalizeHist(gray)
    _, th_eq = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(th_eq)

    return variants

# Perform OCR on processed image variants and return best guess
def read_number_from_image(img):
    variants = preprocess_variants(img)
    guesses = []
    angle = estimate_angle_from_crop(img)
    
    for processed in variants:
        rotated = rotate_image(processed, angle)
        results = reader.readtext(rotated, allowlist="0123456789", detail=1, paragraph=False)
        
        results = [r for r in results if r[2] > 0.5]
        if results:
            guesses.append(results[0][1])

    if not guesses:
        return None

    c = Counter(guesses)
    max_count = max(c.values())
    tied = [num for num, count in c.items() if count == max_count]
    return max(tied, key=lambda x: len(str(x)))

# Compute Levenshtein distance between two strings
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
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[-1][-1]

# Compute normalized similarity score
def similarity(a, b):
    if not a or not b:
        return 0
    dist = levenshtein(a, b)
    return 1 - dist / max(len(str(a)), len(str(b)))

# Compute alignment-based similarity score
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

# Combine similarity metrics into a single score
def combined_score(a, b, w1=0.7, w2=0.3):
    return w1 * similarity(a, b) + w2 * alignment_score(a, b)

# Match detected number to closest valid team number
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

# Define real-world corner coordinates for homography
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
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Error opening video file: {video_path}")

cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame_idx)
ret, calibration_frame = cap.read()
if not ret:
    raise ValueError(f"Could not read frame {calibration_frame_idx}.")

# Extract frame dimensions and initialize view center
H_img, W_img = calibration_frame.shape[:2]
center_x, center_y = W_img // 2, H_img // 2
split_line_x = W_img / 2.0 

# Setup calibration window and mouse interaction
cv2.namedWindow("Calibration Phase", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Calibration Phase", mouse)

print("\n--- DUAL CAMERA CALIBRATION ---")
print("1. Select your 4 reference points on the LEFT field segment.")
print("2. Select the SAME 4 reference points on the RIGHT field segment.")
print("3. Press [ENTER] when all 8 points have been designated.")

# Wait for user to select 8 calibration points
while not calibration_done:
    view, x1, y1 = get_view()
    
    # Draw center split line
    split_view_x = int((split_line_x - x1) / (W_img / zoom) * W_img)
    cv2.line(view, (split_view_x, 0), (split_view_x, H_img), (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw Crosshairs for clicked points
    for i, (px, py) in enumerate(clicked_points):
        sx = int((px - x1) / (W_img / zoom) * W_img)
        sy = int((py - y1) / (H_img / zoom) * H_img)
        
        # Color based on left/right side and RED_LOCATION configuration
        is_left_side = i < 4
        if RED_LOCATION.lower() == 'left':
            color = RED_UI_COLOR if is_left_side else BLUE_UI_COLOR
        else:
            color = BLUE_UI_COLOR if is_left_side else RED_UI_COLOR
        
        ch_size = 12 # Crosshair length
        cv2.line(view, (sx - ch_size, sy), (sx + ch_size, sy), color, 2, cv2.LINE_AA)
        cv2.line(view, (sx, sy - ch_size), (sx, sy + ch_size), color, 2, cv2.LINE_AA)
        cv2.circle(view, (sx, sy), 2, (255, 255, 255), -1, cv2.LINE_AA) # Center dot
        
        label = str(i + 1) if i < 4 else str(i - 3)
        cv2.putText(view, label, (sx + 8, sy - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # UI Overlay Top Banner
    ui_height = 90
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (W_img, ui_height), (15, 15, 15), -1)
    cv2.line(overlay, (0, ui_height), (W_img, ui_height), (0, 140, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, view, 0.15, 0, view)

    # Dynamic UI Text (Using dynamically calculated status colors)
    if len(clicked_points) < 4:
        title_text = "PHASE 1: LEFT FIELD CALIBRATION"
        sub_text = f"Select point {len(clicked_points) + 1} of 4 on the LEFT camera feed."
        status_color = RED_UI_COLOR if RED_LOCATION.lower() == 'left' else BLUE_UI_COLOR
    elif len(clicked_points) < 8:
        title_text = "PHASE 2: RIGHT FIELD CALIBRATION"
        sub_text = f"Select point {len(clicked_points) - 3} of 4 on the RIGHT camera feed."
        status_color = BLUE_UI_COLOR if RED_LOCATION.lower() == 'left' else RED_UI_COLOR
    else:
        title_text = "CALIBRATION COMPLETE"
        sub_text = "Press [ENTER] on your keyboard to finalize and begin tracking."
        status_color = (255, 255, 255)

    cv2.putText(view, title_text, (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 1, cv2.LINE_AA)
    cv2.putText(view, sub_text, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Calibration Phase", view)
    
    # Keyboard listener
    key = cv2.waitKey(1) & 0xFF
    if key in [10, 13] and len(clicked_points) == 8: # Enter key
        calibration_done = True
    elif key == 27: # ESC key to cleanly quit if needed
        print("Calibration cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

if len(clicked_points) != 8:
    raise ValueError(f"You specified {len(clicked_points)} points. You must provide exactly 8 points before continuing.")

pts_left = np.array(clicked_points[:4], dtype=np.float32)
pts_right = np.array(clicked_points[4:], dtype=np.float32)

H_left, _ = cv2.findHomography(pts_left, world_corners_local)
H_right, _ = cv2.findHomography(pts_right, world_corners_local)

print("\nLeft and Right Homography matrices computed successfully.\n")

# Function to draw the field overlay on the top-down map
def draw_field_overlay(canvas, scale, width, height):
    line_color = (255, 255, 255) # White lines for field markings
    thickness = 2

    # Helper function to convert real-world feet coordinates to pixel coordinates on the canvas
    def to_px(x_ft, y_ft):
        px_x = int(x_ft * scale) + (width // 2)
        px_y = int(y_ft * scale) + (height // 2) 
        return (px_x, px_y)

    # Define field perimeter points based on the known dimensions and the corner offsets
    wall_x_right = 0.95 + 27.7297
    wall_y_max = 18.1037 / 2.0
    
    # The corners are located at a 54 degree angle from the vertical, and are 7.0685 ft from the wall
    dx = 7.0685 * np.sin(np.radians(54))
    dy = 7.0685 * np.cos(np.radians(54))
    
    # Calculate corner coordinates based on wall coordinates and offsets
    corner_x_right = wall_x_right - dx
    corner_y_max = wall_y_max + dy
    
    # Mirror the right side coordinates to get the left side coordinates
    wall_x_left = -wall_x_right
    corner_x_left = -corner_x_right

    # Define the perimeter points in a clockwise order starting from the top-left corner
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
    
    # Draw the perimeter
    perimeter_px = np.array([to_px(x, y) for x, y in perimeter_pts_ft], np.int32)
    cv2.polylines(canvas, [perimeter_px], isClosed=True, color=line_color, thickness=thickness)

    # Define the dimensions of the reef
    hex_side = 3.15094154
    hex_dx = hex_side * (np.sqrt(3) / 2.0)
    hex_dy = hex_side / 2.0
    
    right_hex_center_x = 0.95 + 13.0574
    left_hex_center_x = -right_hex_center_x
    
    # Draw the two hexagonal reefs
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

# Reset video to the beginning for tracking phase
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

map_scale = 15
map_width, map_height = 1200, 600

trajectories = defaultdict(list)

print("Starting video tracking from frame 0...")

frame_count = 0

# Main loop to process video frames, perform tracking, OCR, homography mapping, and visualization
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    if not SHOW_VIDEO_FEED and frame_count % 50 == 0:
        print(f"Processed frame {frame_count}...")

    # Track robots in the current frame using the loaded model and custom tracker configuration
    results = robot_model.track(frame, persist=True, tracker=tracker_config, classes=ROBOT_CLASS_IDS, verbose=False)
    
    # Initialize the top-down canvas for this frame if video feed is enabled. This will be used to draw the live map with robot positions.
    if SHOW_VIDEO_FEED:
        top_down_canvas = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        draw_field_overlay(top_down_canvas, map_scale, map_width, map_height)

    # Process each detected robot box and its associated track ID
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # For each detected robot, perform OCR-based team number identification every OCR_INTERVAL frames, and then apply homography to map the position to the top-down view.
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            
            if frame_count % OCR_INTERVAL == 0:
                current_top_votes = max(id_votes[track_id].values()) if id_votes[track_id] else 0
            
                if current_top_votes < 10:
                    crop_y1, crop_y2 = max(0, int(y1)), min(frame.shape[0], int(y2))
                    crop_x1, crop_x2 = max(0, int(x1)), min(frame.shape[1], int(x2))
                    
                    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                        robot_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        number_results = number_model(robot_crop, verbose=False)[0]
                        
                        for nbox in number_results.boxes:
                            cls_id = int(nbox.cls[0])
                            
                            if cls_id == RED_NUMBER_CLASS_ID:
                                team_list = RED_TEAMS
                            elif cls_id == BLUE_NUMBER_CLASS_ID:
                                team_list = BLUE_TEAMS
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

                            if matched is not None:
                                id_votes[track_id][matched] += 1
                                    
            if id_votes[track_id]:
                assigned_team = max(id_votes[track_id], key=id_votes[track_id].get)
                DYNAMIC_ID_TO_TEAM[track_id] = assigned_team
            
            display_id = DYNAMIC_ID_TO_TEAM.get(track_id, f"ID:{track_id}")
            
            # Calculate the bottom-center point of the bounding box for homography mapping
            bc_x = (x1 + x2) / 2.0
            bc_y = y2
            
            # Visualize the tracking box and ID on the original video feed if enabled
            if SHOW_VIDEO_FEED:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(bc_x), int(bc_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(display_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            pts = np.array([[[bc_x, bc_y]]], dtype=np.float32)
            
            # Determine which homography to use based on the x-coordinate of the bottom-center point, then transform to top-down coordinates and convert to feet
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
            
            # Convert real-world feet coordinates to pixel coordinates on the top-down map canvas
            map_x = int(real_x_ft * map_scale) + (map_width // 2)
            map_y = int(real_y_ft * map_scale) + (map_height // 2)
            
            # Only consider points that fall within the map boundaries to avoid errors and ensure valid trajectory plotting
            if 0 <= map_x < map_width and 0 <= map_y < map_height:
                if SHOW_VIDEO_FEED:
                    dot_color = get_team_color(display_id) if isinstance(display_id, int) else (0, 255, 255)
                    cv2.circle(top_down_canvas, (map_x, map_y), 10, dot_color, -1)
                    cv2.putText(top_down_canvas, str(display_id), (map_x + 15, map_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                trajectories[track_id].append((map_x, map_y))
    
    # Draw the trajectories for all tracked robots on the top-down canvas if video feed is enabled
    if SHOW_VIDEO_FEED:
        cv2.imshow("Original Video with Tracking", frame)
        cv2.imshow("Top-Down Live Map (Feet)", top_down_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Video processing complete. Generating final trajectory map...")

# Create the final map canvas and draw the field overlay. Then, plot all trajectories with a layered approach to ensure visibility of both mapped and unmapped paths, and add a static legend for team colors.
final_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
draw_field_overlay(final_map, map_scale, map_width, map_height)

mapped_segments = defaultdict(list)
unmapped_segments = []

# Separate trajectories into mapped (with known team) and unmapped (unknown team) for layered visualization. This ensures that all paths are visible, with mapped paths highlighted in their team colors and unmapped paths shown in a neutral color.
for track_id, points in trajectories.items():
    if len(points) < 2: 
        continue 
    
    team = DYNAMIC_ID_TO_TEAM.get(track_id)
    if team in ALL_TEAMS:
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

for team in ALL_TEAMS:
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

if RED_LOCATION.lower() == 'left':
    cv2.putText(final_map, "RED ALLIANCE", (left_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, RED_UI_COLOR, 2)
    cv2.putText(final_map, "BLUE ALLIANCE", (right_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, BLUE_UI_COLOR, 2)
else:
    cv2.putText(final_map, "BLUE ALLIANCE", (left_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, BLUE_UI_COLOR, 2)
    cv2.putText(final_map, "RED ALLIANCE", (right_label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, RED_UI_COLOR, 2)

base_name, ext = os.path.splitext(image_output_name)
counter = 1
final_output_name = image_output_name
while os.path.exists(final_output_name):
    final_output_name = f"{base_name} ({counter}){ext}"
    counter += 1

cv2.imshow("Final Trajectories - Press any key to close", final_map)
cv2.imwrite(final_output_name, final_map) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
print(f"Final map saved as '{final_output_name}'. Process finished.")