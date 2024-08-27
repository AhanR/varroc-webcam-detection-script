import cv2
import threading
from ultralytics import YOLOv10
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import logging
import os
from dotenv import load_dotenv
import time
import numpy as np

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

load_dotenv("config.env")

CAN_FRAME_ID = 0x0A
SHOW_BBOX = True if os.getenv("SHOW_BOXES") == "True" else False
CONFIDENCE_CUTOFF = float(os.getenv("CONFIDENCE_FILTER"))
COORDS_PERCENTAGE = True if os.getenv("COORDS_PERCENTAGE") == "True" else False
FILTER_OBJECTS = True if os.getenv("FILTER_OBJECTS") == "True" else False
CANVAS_WIDTH = int(os.getenv("CANVAS_WIDTH"))
CANVAS_HEIGHT = int(os.getenv("CANVAS_HEIGHT"))

FILTER_OBJECT_IDS = [0,1,2,3,5,7,9,11,12,14,15,16,17,18,19,20,21,22,23,67,79]

# Function to detect objects using YOLOv10 and annotate the frame
def detect_objects(frame, model):
    results = model(frame, stream=True)
    detected_objects = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
    
        # Initialize ImageDraw object
        draw = ImageDraw.Draw(canvas)
        


        for box, score, cls in zip(boxes, scores, classes):

            detected_objects.append((box, int(cls), score))
            x1, y1, x2, y2 = map(int, box)
            
            # Draw a black rectangle on the canvas
            draw.rectangle([(x1,y1),(x2,y2)], fill='black')

            # send can data from here
            print(f"Timestamp: {time.time()} {{'box': {box}, 'class': {int(cls)}, 'score': {score:.2f}}}")

            # Filter for
            # Vehicle
            # Human
            # Animal
            # Sign boards
            # pen
            # toothbrush
            # mobile

            if SHOW_BBOX and score > CONFIDENCE_CUTOFF and int(cls) in FILTER_OBJECT_IDS:
                # Draw bounding box and class ID on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{int(cls)}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # add script to send data to CANoe

    return frame, canvas

# Thread for displaying video
def display_video(video_source, stop_event, frame_lock, shared_frame, label, label_2):
    cap = cv2.VideoCapture(video_source)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            shared_frame[0] = frame
        if stop_event.is_set():
            break
        time.sleep(1/int(os.getenv("VIDEO_FPS")))
    cap.release()

# Thread for processing frames
def process_frames(stop_event, frame_lock, shared_frame, model, label, label_2, root):
    while not stop_event.is_set():
        with frame_lock:
            frame = shared_frame[0].copy() if shared_frame[0] is not None else None
        if frame is not None:
            annotated_frame, bw_frame = detect_objects(frame, model)

            # Convert annotated frame to ImageTk format
            image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # Use after to update the UI from the main thread
            root.after(0, update_ui, label, image, label_2, bw_frame, root)

def update_ui(label, image, label_2, bw_frame, root):
    # Update the label with the new image
    label.config(image=image)
    label.image = image

    bw_frame = ImageTk.PhotoImage(bw_frame)
    label_2.config(image=bw_frame)
    label_2.image = bw_frame


def on_close(stop_event, root):
    stop_event.set()
    root.destroy()

def start_video(video_path, stop_event, frame_lock, shared_frame, video_label, model, video_label_2, root):
    stop_event.clear()
    display_thread = threading.Thread(target=display_video, args=(video_path, stop_event, frame_lock, shared_frame, video_label, video_label_2))
    process_thread = threading.Thread(target=process_frames, args=(stop_event, frame_lock, shared_frame, model, video_label, video_label_2, root))

    display_thread.start()
    process_thread.start()

    return display_thread, process_thread

def main(video_path):

    global FILTER_OBJECT_IDS

    model = YOLOv10.from_pretrained(os.getenv("MODEL"))
    stop_event = threading.Event()
    frame_lock = threading.Lock()
    shared_frame = [None]

    if not FILTER_OBJECTS:
        FILTER_OBJECT_IDS = list(model.names.keys())

    root = tk.Tk()
    root.title("YOLOv10 Video Processing")

    video_label = tk.Label(root)
    video_label.pack()

    control_frame = tk.Frame(root)
    control_frame.pack()

    independent_frame = tk.Toplevel(root)
    independent_frame.title("Intermediate window")

    intermediate_lablel = tk.Label(independent_frame)
    intermediate_lablel.pack()

    display_thread = None
    process_thread = None

    def start():
        nonlocal display_thread, process_thread
        if display_thread is None or not display_thread.is_alive():
            display_thread, process_thread = start_video(video_path, stop_event, frame_lock, shared_frame, video_label, model, intermediate_lablel, root)

    def stop():
        stop_event.set()
        if display_thread is not None:
            display_thread.join()
        if process_thread is not None:
            process_thread.join()

    def switch_to_webcam():
        nonlocal display_thread, process_thread
        stop()
        display_thread, process_thread = start_video(0, stop_event, frame_lock, shared_frame, video_label, model, intermediate_lablel)

    start_button = tk.Button(control_frame, text="Start", command=start)
    start_button.pack(side=tk.LEFT)

    stop_button = tk.Button(control_frame, text="Stop", command=stop)
    stop_button.pack(side=tk.LEFT)

    webcam_button = tk.Button(control_frame, text="Webcam", command=switch_to_webcam)
    webcam_button.pack(side=tk.LEFT)

    close_button = tk.Button(control_frame, text="Close", command=lambda: on_close(stop_event, root))
    close_button.pack(side=tk.LEFT)

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(stop_event, root))
    root.mainloop()

    if display_thread is not None:
        display_thread.join()
    if process_thread is not None:
        process_thread.join()

if __name__ == "__main__":
    video_path = os.getenv("VIDEO_FILE")  # Replace with your video file path
    # video_path = r"C:\Users\ahan6\Downloads\VID-20240711-WA0008.mp4"  # Replace with your video file path
    main(video_path)
