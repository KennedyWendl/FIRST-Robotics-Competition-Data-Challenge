import cv2
import os

video = "final2.mp4"
out_dir = "trainingImages"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video)
fps = int(cap.get(cv2.CAP_PROP_FPS))

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % fps == 0:
        height, width, _ = frame.shape

        # Crop bottom half
        bottom_half = frame[3*height//5:height, 0:width]

        cv2.imwrite(f"{out_dir}/frame{video[:-4]}_{saved:04d}.jpg", bottom_half)
        saved += 1

    count += 1

cap.release()
print(f"Saved {saved} cropped frames")