import cv2

# Function to list available video capture devices
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# List available cameras
cameras = list_cameras()

if cameras:
    print("Available cameras:")
    for camera in cameras:
        print(f"Camera index: {camera}")
else:
    print("No cameras found.")
