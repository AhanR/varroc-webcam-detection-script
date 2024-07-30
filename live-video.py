import cv2
import threading
import logging
from threading import Event, Lock
from ultralytics import YOLOv10

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Function to detect objects using YOLOv10
def detect_objects(frame, model):
    results = model(frame, stream=True)
    detected_objects = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            detected_objects.append((box, int(cls), score))
            print(f"{{'box': {box}, 'class': {int(cls)}, 'score': {score:.2f}}}")

    return detected_objects

# Thread for displaying video
def display_video(video_path, stop_event : Event, frame_lock : Lock, shared_frame):
    cap = cv2.VideoCapture(video_path)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            shared_frame[0] = frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()

# Thread for processing frames
def process_frames(stop_event, frame_lock, shared_frame, model):
    while not stop_event.is_set():
        with frame_lock:
            frame = shared_frame[0].copy() if shared_frame[0] is not None else None
        if frame is not None:
            detect_objects(frame, model)

def main(video_path):
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    stop_event = threading.Event()
    frame_lock = threading.Lock()
    shared_frame = [None]

    print("Ready player one")

    display_thread = threading.Thread(target=display_video, args=(video_path, stop_event, frame_lock, shared_frame))
    process_thread = threading.Thread(target=process_frames, args=(stop_event, frame_lock, shared_frame, model))

    display_thread.start()
    process_thread.start()

    display_thread.join()
    process_thread.join()

if __name__ == "__main__":
    video_path = r"C:\Users\ahan6\Downloads\VID-20240711-WA0008.mp4"  # Replace with your video file path
    main(video_path)
