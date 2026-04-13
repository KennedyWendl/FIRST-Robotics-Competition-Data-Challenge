import os
import re
import cv2
import pandas as pd
from yt_dlp import YoutubeDL
import imageio_ffmpeg
from ultralytics import YOLO
import easyocr


def process_youtube_video(
    youtube_link,
    videos_path="videos",
    crop_model_path="models/crop_scoreboard.pt",
    info_model_path="models/extract_scoreboard_info.pt",
    frame_skip=15,
    device=0,
    delete_video=False
):
    # --- Setup ---
    os.makedirs(videos_path, exist_ok=True)

    crop_model = YOLO(crop_model_path)
    info_model = YOLO(info_model_path)
    reader = easyocr.Reader(['en'], gpu=(device != "cpu"))

    # --- Helper functions ---
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

    def read_number(img):
        images = preprocess_for_ocr(img)
        text = read_best_text(images, '0123456789')

        if text is None:
            return None

        text = re.sub(r"\D", "", text)
        return int(text) if text else None

    def read_timer(img):
        images = preprocess_for_ocr(img)
        text = read_best_text(images, '0123456789:')

        if text is None:
            return None

        match = re.search(r"\d{1,2}:\d{2}", text)
        return match.group(0) if match else None

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

    # --- Download video ---
    video_filename = download_video(youtube_link, videos_path)
    video_path = os.path.join(videos_path, video_filename)

    # --- Tracking variables ---
    rows = []
    prev_blue = None
    prev_red = None
    pending_row = None
    is_auto = True

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        crop_results = crop_model(frame, device=device, verbose=False)[0]

        if len(crop_results.boxes) == 0:
            frame_idx += 1
            continue

        x1, y1, x2, y2 = map(int, crop_results.boxes.xyxy[0])
        scoreboard = frame[y1:y2, x1:x2]

        info_results = info_model(scoreboard, device=device, verbose=False)[0]

        blue_score = None
        red_score = None
        timer = None
        blue_center_x = None
        red_center_x = None
        team_data = []

        for b in info_results.boxes:
            cls_id = int(b.cls[0])
            label = info_model.names[cls_id]

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            region = scoreboard[y1:y2, x1:x2]

            x_center = (x1 + x2) / 2

            if label == "blue_score":
                blue_score = read_number(region)
                blue_center_x = x_center

            elif label == "red_score":
                red_score = read_number(region)
                red_center_x = x_center

            elif label == "timer":
                timer = read_timer(region)

            elif label == "team_number":
                num = read_number(region)
                if num is not None:
                    team_data.append((num, x_center))

        if blue_score is None and red_score is None and timer is None:
            frame_idx += 1
            continue

        if prev_blue is not None and blue_score is not None:
            if blue_score < prev_blue:
                frame_idx += 1
                continue

        if prev_red is not None and red_score is not None:
            if red_score < prev_red:
                frame_idx += 1
                continue

        if prev_blue == blue_score and prev_red == red_score:
            frame_idx += 1
            continue

        if timer is not None and is_auto:
            try:
                minutes = int(timer.split(":")[0])
                if minutes == 2:
                    is_auto = False
            except:
                pass

        red_location = None
        if blue_center_x is not None and red_center_x is not None:
            red_location = "right" if red_center_x > blue_center_x else "left"

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

        blue_team_numbers = [n for n, _ in sorted(blue_teams, key=lambda x: x[1])[:3]]
        red_team_numbers = [n for n, _ in sorted(red_teams, key=lambda x: x[1])[:3]]

        current_row = {
            "Frame": frame_idx,
            "blue_score": blue_score,
            "red_score": red_score,
            "timer": timer,
            "is_auto": is_auto,
            "red_location": red_location,
            "blue_team_numbers": blue_team_numbers,
            "red_team_numbers": red_team_numbers,
            "youtube_link": youtube_link,
        }

        if blue_score is not None and red_score is not None and timer is None:
            if pending_row is None:
                pending_row = current_row
            frame_idx += 1
            continue

        if (
            pending_row is not None and
            timer is not None and
            blue_score == pending_row["blue_score"] and
            red_score == pending_row["red_score"]
        ):
            rows.append(current_row)
            pending_row = None

        else:
            if pending_row is not None:
                rows.append(pending_row)
                pending_row = None

            rows.append(current_row)

        prev_blue = blue_score
        prev_red = red_score

        frame_idx += 1

    if pending_row is not None:
        rows.append(pending_row)

    cap.release()

    if delete_video and os.path.exists(video_path):
        os.remove(video_path)

    return pd.DataFrame(rows)