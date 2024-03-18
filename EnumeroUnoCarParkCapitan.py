import argparse
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import torch

print(torch.cuda.is_available())

# Define the set of vehicle-related class IDs
vehicle_classes = {1,2,3,4,5,6,7,8}  # For COCO dataset: 2 = car, 5 = bus, 7 = truck

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process line coordinates.')
parser.add_argument('--lines', type=str, required=True, help='Path to a YAML file containing the line coordinates.')
parser.add_argument('--top', type=int, default=0, help='Percentage of the top region to ignore.')
parser.add_argument('--bottom', type=int, default=0, help='Percentage of the bottom region to ignore.')
parser.add_argument('--left', type=int, default=0, help='Percentage of the left region to ignore.')
parser.add_argument('--right', type=int, default=0, help='Percentage of the right region to ignore.')
args = parser.parse_args()

# Load the line coordinates from the YAML file
with open(args.lines, 'r') as f:
    lines = yaml.safe_load(f)

# Convert the list of dictionaries to a dictionary for easier access
lines_dict = {line['out' if 'out' in line else 'in']: line for line in lines}

# Initialize the YOLO model
model = YOLO('yolov8n.pt')  # Ensure the model name matches your file

# Open the video file
video_path = '/media/glen/Linux/e.mp4'  # Update the path to the video file video_path = 'd.mp4'/media/glen/Linux
cap = cv2.VideoCapture(video_path)

# Get the width and height of the frame
_, frame = cap.read()
frame_width = frame.shape[1]
frame_height = frame.shape[0]

# Calculate the regions to ignore
x1_ignore = int(args.left / 100 * frame_width)
x2_ignore = frame_width - int(args.right / 100 * frame_width)
y1_ignore = int(args.top / 100 * frame_height)
y2_ignore = frame_height - int(args.bottom / 100 * frame_height)

# Initialize dictionaries to keep track of IDs of vehicles that have crossed the lines and the counts
crossed_ids = {line_key: set() for line_key in lines_dict}
counts = {line_key: {'car': 0} for line_key in lines_dict}

# Initialize dictionaries to keep track of the previous positions of the vehicles
prev_positions = {}

# Initialize dictionaries to keep track of the vehicles that are currently in the counting regions
in_regions = {line_key: set() for line_key in lines_dict}

frame_count = 0
scale_factor = 1  # Adjust this to match the factor you used to resize the frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to half its original size
    frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))

    # Mask the regions to ignore
    frame[:y1_ignore, :] = 0
    frame[y2_ignore:, :] = 0
    frame[:, :x1_ignore] = 0
    frame[:, x2_ignore:] = 0

    frame_count += 1
    if frame_count % 100 == 0:  # Skip every second frame
        continue

    # Process the frame with the YOLO model
    results = model.track(frame, persist=True, conf=0.8)

    # Check if results is not empty
    if results:
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get the class ID
                class_id = int(box.cls)

                # Check if the class ID of the detection is in the set of vehicle-related class IDs
                if class_id not in vehicle_classes:
                    continue

                # Rest of your code...
                track_id = getattr(box, 'id', [None])
                if track_id is not None:
                    track_id = track_id[0]
                else:
                    track_id = None

                if track_id is not None:
                    track_id = int(track_id)
                    prev_position = prev_positions.get(track_id)

                    # Compute the centroid of the bounding box
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2

                    # Use the line coordinates from the YAML file
                    for line_key in lines_dict:
                        line = lines_dict[line_key]

                        # Scale the line coordinates
                        line_x1, line_y1, line_x2, line_y2 = map(lambda x: int(x / scale_factor), [line["x1"], line["y1"], line["x2"], line["y2"]])

                        # Compute the cross product of the vectors formed by the line and the positions of the vehicle
                        cross_product1 = (centroid_x - line_x1) * (line_y2 - line_y1) - (centroid_y - line_y1) * (line_x2 - line_x1)

                        if prev_position is not None:
                            prev_x, prev_y = prev_position
                            cross_product2 = (prev_x - line_x1) * (line_y2 - line_y1) - (prev_y - line_y1) * (line_x2 - line_x1)

                            # Check if the vehicle has crossed the line
                            if cross_product1 * cross_product2 < 0 and track_id not in crossed_ids[line_key]:
                                crossed_ids[line_key].add(track_id)
                                counts[line_key]['car'] += 1

                    # Draw bounding box and label
                    label = f"ID: {track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # Update the previous position of the vehicle
                    prev_positions[track_id] = (centroid_x, centroid_y)

    # Remove the IDs of the vehicles that are no longer in the counting regions
    for line_key in in_regions:
        in_regions[line_key] = in_regions[line_key].intersection(set(prev_positions.keys()))

    # Draw the counting lines and the counting regions
    for line_key in lines_dict:
        line = lines_dict[line_key]

        # Scale the line coordinates
        line_x1, line_y1, line_x2, line_y2 = map(lambda x: int(x / scale_factor), [line["x1"], line["y1"], line["x2"], line["y2"]])

        cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (line_x1 - 10, line_y1 - 10), (line_x2 + 10, line_y2 + 10), (255, 0, 0), 2)
        cv2.rectangle(frame, (line_x1 - 10, line_y1 + 10), (line_x2 + 10, line_y2 - 10), (255, 0, 0), 2)

    # Display counts for cars
    x_offset = 10
    y_offset = 60
    text_scale = 0.8  # Increase the text scale

    for i, line_key in enumerate(lines_dict):
        label = f'Car Count {line_key}: {counts[line_key]["car"]}'
        cv2.putText(frame, label, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)
        y_offset += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)[0][1] + 10

    # Display the frame
    cv2.imshow("YOLO Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print summary of counts
for line_key in lines_dict:
    print(f"Total Car Count {line_key}: {counts[line_key]['car']}")