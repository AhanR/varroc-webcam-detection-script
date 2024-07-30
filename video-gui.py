import cv2
import threading
import time
from ultralytics import YOLOv10
import tkinter as tk
from PIL import Image, ImageTk

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
def display_video(video_path, stop_event, frame_lock, shared_frame, label):
    cap = cv2.VideoCapture(video_path)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            shared_frame[0] = frame
        # Convert frame to ImageTk format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image
        if stop_event.is_set():
            break
    cap.release()

# Thread for processing frames
def process_frames(stop_event, frame_lock, shared_frame, model):
    while not stop_event.is_set():
        with frame_lock:
            frame = shared_frame[0].copy() if shared_frame[0] is not None else None
        if frame is not None:
            detect_objects(frame, model)

def on_close(stop_event, root):
    stop_event.set()
    root.destroy()

def main(video_path):
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    stop_event = threading.Event()
    frame_lock = threading.Lock()
    shared_frame = [None]

    root = tk.Tk()
    root.title("YOLOv10 Video Processing")

    video_label = tk.Label(root)
    video_label.pack()

    control_frame = tk.Frame(root)
    control_frame.pack()

    close_button = tk.Button(control_frame, text="Close Video", command=lambda: on_close(stop_event, root))
    close_button.pack()

    print("ready player")

    display_thread = threading.Thread(target=display_video, args=(video_path, stop_event, frame_lock, shared_frame, video_label))
    process_thread = threading.Thread(target=process_frames, args=(stop_event, frame_lock, shared_frame, model))

    display_thread.start()
    process_thread.start()

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(stop_event, root))
    root.mainloop()

    display_thread.join()
    process_thread.join()

if __name__ == "__main__":
    video_path = r"C:\Users\ahan6\Downloads\VID-20240711-WA0008.mp4"  # Replace with your video file path
    main(video_path)
