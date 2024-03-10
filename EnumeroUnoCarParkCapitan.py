import cv2
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8l.pt')  # Ensure the model name matches your file

# Open the video file
video_path = '/media/glen/Linux/d.mp4'  # Update the path to the video file video_path = 'd.mp4'
cap = cv2.VideoCapture(video_path)

# Get the width of the frame
_, frame = cap.read()
frame_width = frame.shape[1]

# Define the lines for counting
line1_x1, line1_x2, line1_y = 350, 450, 530
line2_x1, line2_x2, line2_y = line1_x1 - 100, line1_x2 - 100, line1_y - 100  # Moved down by 100 pixels

# Adjust the coordinates for the third line
line3_x = line1_x1 + 380 + 30  # Moved right by 380 pixels and further right by 30 pixels
line3_y1 = line1_y - 50 - 70  # Moved up by 50 pixels and further up by 70 pixels
line3_y2 = line1_y + 50 - 70  # Moved up by 50 pixels and further up by 70 pixels

# Initialize sets to keep track of IDs of vehicles that have crossed the lines
crossed_line1_ids, crossed_line3_ids = set(), set()
crossed_line1_to_2_ids, crossed_line3_to_2_ids = set(), set()

# Initialize dictionaries to keep track of the previous positions of the vehicles
prev_positions = {}

# Initialize counters for cars, trucks, and buses
car_count_1_to_2, truck_count_1_to_2, bus_count_1_to_2 = 0, 0, 0
car_count_3_to_2, truck_count_3_to_2, bus_count_3_to_2 = 0, 0, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with the YOLO model
    results = model.track(frame, persist=True, conf=0.3, iou=0.5)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = getattr(box, 'id', [None])[0]
            cls_name = result.names[int(box.cls[0])].lower()

            if track_id is not None:
                track_id = int(track_id)
                prev_position = prev_positions.get(track_id)

                # Check if the vehicle has crossed the first or third line
                if ((x1 <= line1_x1 <= x2 or x1 <= line1_x2 <= x2) and 
                    (y1 <= line1_y < y2 or y2 < line1_y <= y1)):
                    crossed_line1_ids.add(track_id)
                elif prev_position is not None and prev_position[1] > line3_y1 and y1 <= line3_y1:
                    crossed_line3_ids.add(track_id)

                # Check if the vehicle has crossed the second line and was previously detected crossing the first or third line
                if ((x1 <= line2_x1 <= x2 or x1 <= line2_x2 <= x2) and 
                    (y1 <= line2_y < y2 or y2 < line2_y <= y1)):
                    if track_id in crossed_line1_ids and track_id not in crossed_line1_to_2_ids:
                        crossed_line1_to_2_ids.add(track_id)  # Mark the vehicle as counted
                        if cls_name == 'car':
                            car_count_1_to_2 += 1
                        elif cls_name == 'truck' or cls_name == 'train':  # Include 'train' in the truck count
                            truck_count_1_to_2 += 1
                            cls_name = 'truck'  # Change the label to 'truck'
                        elif cls_name == 'bus':
                            bus_count_1_to_2 += 1
                    elif track_id in crossed_line3_ids and track_id not in crossed_line3_to_2_ids and track_id not in crossed_line1_ids:
                        crossed_line3_to_2_ids.add(track_id)  # Mark the vehicle as counted
                        if cls_name == 'car':
                            car_count_3_to_2 += 1
                        elif cls_name == 'truck' or cls_name == 'train':  # Include 'train' in the truck count
                            truck_count_3_to_2 += 1
                            cls_name = 'truck'  # Change the label to 'truck'
                        elif cls_name == 'bus':
                            bus_count_3_to_2 += 1

                # Draw bounding box and label
                label = f"{cls_name} {box.conf[0]:.2f} ID: {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Update the previous position of the vehicle
                prev_positions[track_id] = (x1, y1)

    # Draw the counting lines
    cv2.line(frame, (line1_x1, line1_y), (line1_x2, line1_y), (255, 0, 0), 2)
    cv2.line(frame, (line2_x1, line2_y), (line2_x2, line2_y), (255, 0, 0), 2)
    cv2.line(frame, (line3_x, line3_y1), (line3_x, line3_y2), (255, 0, 0), 2)

    # Display counts for cars, trucks, and buses
    cv2.putText(frame, f'Car Count 1->2: {car_count_1_to_2}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Truck Count 1->2: {truck_count_1_to_2}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Bus Count 1->2: {bus_count_1_to_2}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f'Car Count 3->2: {car_count_3_to_2}', (frame_width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Truck Count 3->2: {truck_count_3_to_2}', (frame_width - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Bus Count 3->2: {bus_count_3_to_2}', (frame_width - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("YOLO Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print summary of counts
print(f"Total Car Count 1->2: {car_count_1_to_2}, 3->2: {car_count_3_to_2}")
print(f"Total Truck Count 1->2: {truck_count_1_to_2}, 3->2: {truck_count_3_to_2}")
print(f"Total Bus Count 1->2: {bus_count_1_to_2}, 3->2: {bus_count_3_to_2}")
