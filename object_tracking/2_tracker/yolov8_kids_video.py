from ultralytics import YOLO
import cv2
import csv
import torch
import numpy as np

model_path = '/.../best.pt'  # yolov8s8000
video_path = '/.../File.mp4'
output_path = '/.../File_yolov8s8000.mp4'
csv_path = '/.../File_yolov8s8000.csv'

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(model_path)
model.to(device)

# Video capture and output setup
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Original width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Original height
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Use original resolution

# CSV file setup
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Timestamp(ms)']
    max_objects = 8
    for i in range(max_objects):
        header.extend([f'Object_ID_{i}', f'Class_{i}', f'Center_X_{i}', f'Center_Y_{i}', f'Width_{i}', f'Height_{i}', f'Confidence_{i}'])
    writer.writerow(header)

    previous_timestamp = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if previous_timestamp is not None:
                print(f"Interval: {timestamp - previous_timestamp} ms")
            previous_timestamp = timestamp

            # Object detection and tracking on GPU
            results = model.track(frame, persist=True, device=device)  

            # Visualization (plot detections and return as numpy array)
            frame_ = results[0].plot()

            # CSV file
            row_data = [timestamp]  
            object_count = 0

            # detected objects
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for box in results[0].boxes:
                    if object_count >= max_objects:
                        break  
                    
                    xywh = box.xywh.cpu().numpy().flatten()  # Center X, Center Y, Width, Height
                    conf = box.conf.cpu().item()  # Confidence score
                    cls = int(box.cls.cpu().item())  # Class index
                    obj_id = int(box.id.cpu().item()) if box.id is not None else ''  # Object ID (if available)

                    # Append object details to the row
                    row_data.extend([obj_id, model.names[cls], xywh[0], xywh[1], xywh[2], xywh[3], conf])
                    object_count += 1

            # remaining columns for undetected objects
            while len(row_data) < len(header):
                row_data.extend(['', '', '', '', '', '', ''])

            # Write the row to the CSV
            writer.writerow(row_data)

            frame_ = frame_.astype(np.uint8)  # correct data type
            frame_ = np.ascontiguousarray(frame_)  # proper memory alignment

            # Write the frame to output with the original resolution
            out.write(frame_)

            # frame display
            cv2.imshow('Frame', frame_)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


