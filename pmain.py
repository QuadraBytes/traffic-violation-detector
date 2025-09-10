import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
from test1 import process_frame
import os
from tracker import*
from datetime import datetime


# Load YOLO model
model = YOLO("yolov10s.pt")

# Read class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

# Initialize tracker
tracker = Tracker()

# Area polygon for violation detection
area = [(324, 313), (283, 374), (854, 392), (864, 322)]

# Create directory for today's date
today_date = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('saved_images', today_date)
os.makedirs(output_dir, exist_ok=True)

# Track IDs already saved
saved_ids = set()

# Open video
cap = cv2.VideoCapture('tr.mp4')
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # Skip every other frame for speed
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    # Process frame for light color
    processed_frame, detected_label = process_frame(frame)
    # Run YOLO detection
    results = model(frame)
    boxes = results[0].boxes.data
    if len(boxes) == 0:
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    px = pd.DataFrame(boxes).astype("float")
    detected_list = []
    for _, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        class_name = class_list[class_id] if class_id < len(class_list) else str(class_id)
        detected_list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(detected_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        # Only check for cars inside the area
        class_name = None
        for det in px.itertuples():
            if int(det[1]) == x3 and int(det[2]) == y3 and int(det[3]) == x4 and int(det[4]) == y4:
                class_id = int(det[6])
                class_name = class_list[class_id] if class_id < len(class_list) else str(class_id)
                break
        if result >= 0:
            if class_name and 'car' in class_name.lower() and detected_label == "RED":
                cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                # Save image if not already saved for this ID
                if obj_id not in saved_ids:
                    saved_ids.add(obj_id)
                    timestamp = datetime.now().strftime('%H-%M-%S')
                    image_filename = f"{timestamp}_{obj_id}.jpg"
                    output_path = os.path.join(output_dir, image_filename)
                    cv2.imwrite(output_path, frame)
            else:
                cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
