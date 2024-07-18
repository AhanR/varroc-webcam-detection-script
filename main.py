import cv2
from ultralytics import YOLOv10
import logging
import time
import keyboard

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

model_size = input("Pleae enter a model size from the following list:\nNano (n)\nsmall (s)\nMedium (m)\nBig (b)\nLarge (l)\nExtra Large (x)\nChoose an option {n/s/m/b/l/x}:").strip()
if model_size not in ("nsmblx"):
    print(f"Could not find size: {model_size}. "+"Input must be {n/s/m/b/l/x}, single letter input only.\nTerminating program.")
    exit(0)

# 0 usually reffers to the webcam but can differ based on the system, run cameras.py to get the list of cameras and preview.py to check the input signal from the cameras
def list_cameras():
    index = 0
    arr = []
    try:
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
    except:
        arr = []
    return arr

cam_list = list_cameras()
if cam_list == []:
    print("No cameras found, exiting the program.")
    exit()
camera_index = int(input("Select which video source you want to use from the list below:\n"+"\n".join(map(lambda x: 'camera ('+str(x) + ')', cam_list))+ "\nChoose and option {"+"/".join(map(lambda x: str(x), cam_list))+"}:"))
if camera_index not in cam_list:
    print(f"Could not find camera number f{camera_index} in list. Enter a number from the list:", cam_list)

path = input("Enter file path for logs:")

model = YOLOv10.from_pretrained(f'jameslahm/yolov10{model_size}')
print("Key: object dictionary")
print(model.names)
print("Press ESC key to stop program")
print("{[x1, y1, x2, y2], Class_ID, Confidence_Score}")

# Open a connection to the selected webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

t1 = time.time()
n_frames = 0
log_file = open(path, "w")
# Run the video feed loop
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
            label = f"{[int(cls)]}: {score:.2f}"
            print("{",box,",", cls,",", score,"}")
            log_file.write("{"+str(box)+","+str(cls)+","+str(score)+"}\n")

    # Show the frame on the window
    # cv2.imshow("preview", frame)
    n_frames += 1
    # Exit the loop if the 'q' key is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if keyboard.is_pressed('esc'):
        break
t2 = time.time()
log_file.close()
print("Stats:")
print(f"Frames: {n_frames}")
print(f"AVG FPS: {n_frames/(t2-t1)}")
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
