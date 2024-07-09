import cv2
from ultralytics import YOLOv10

# Load the YOLOv5 model
model = YOLOv10.from_pretrained('jameslahm/yolov10x')

# 0 usually reffers to the webcam but can differ based on the system, run cameras.py to get the list of cameras and preview.py to check the input signal from the cameras
camera_index = 0

# Create a named window to show the example
cv2.namedWindow("preview")

# Open a connection to the selected webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Run the video feed loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    # Draw the bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(cls)]}: {score:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame on the window
    cv2.imshow("preview", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
