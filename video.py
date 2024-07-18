import cv2
from ultralytics import YOLOv10
import logging
import time

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Prompt for model size
model_size = input("Please enter a model size from the following list:\nNano (n)\nSmall (s)\nMedium (m)\nBig (b)\nLarge (l)\nExtra Large (x)\nChoose an option {n/s/m/b/l/x}:").strip()
if model_size not in ("nsmblx"):
    print(f"Could not find size: {model_size}. Input must be one of n/s/m/b/l/x, single letter input only.\nTerminating program.")
    exit(0)

# Get input and output video paths
input_video_path = input("Enter the path to the input video file: ").strip()
output_video_path = input("Enter the path to save the output video file: ").strip()

# Load the YOLO model
model = YOLOv10.from_pretrained(f'jameslahm/yolov10{model_size}')

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Key: object dictionary")
print(model.names)
print("Processing video...")

# Run the video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    # Draw the bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(cls)]}: {score:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Write the frame into the output video file
    out.write(frame)

print("Processing completed. Video saved to:", output_video_path)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()