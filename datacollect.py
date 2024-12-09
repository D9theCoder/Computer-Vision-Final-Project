import cv2
import os
import time

# Global variables
video = None
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def initialize_capture(name):
    global video
    video = cv2.VideoCapture(0)
    id_folder = os.path.join('datasets', name)
    os.makedirs(id_folder, exist_ok=True)
    existing_images = [f for f in os.listdir(id_folder) if f.startswith(f'User.{name}.')]
    count = max([int(f.split('.')[2]) for f in existing_images]) if existing_images else 0
    return count

def collect_frame(name, count, last_capture_time=None):
    global video
    if not video or not video.isOpened():
        return None, count, last_capture_time

    ret, frame = video.read()
    if not ret:
        return None, count, last_capture_time

    current_time = time.time()
    frame_captured = False
    
    # Only capture new image if 0.005 seconds have passed since last capture
    if last_capture_time is None or (current_time - last_capture_time) >= 0.005:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f'datasets/{name}/User.{name}.{count}.jpg', gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
            cv2.putText(frame, str(count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
            last_capture_time = current_time
            frame_captured = True
            break

    # Draw rectangle even if we don't capture the image
    if not frame_captured:
        faces = facedetect.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
            cv2.putText(frame, str(count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
            break

    return frame, count, last_capture_time

def release_capture():
    global video
    if video:
        video.release()
        video = None