import cv2
import os
import time
import numpy as np
from skimage.feature import hog
from sklearn.svm import OneClassSVM  # Change import
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import random

# Global variables for video capture and face detection
video = None
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize and release video capture

def initialize_capture(name):
    global video
    video = cv2.VideoCapture(0)
    id_folder = os.path.join('datasets', name)
    os.makedirs(id_folder, exist_ok=True)
    existing_images = [f for f in os.listdir(id_folder) if f.startswith(f'User.{name}.')]
    count = max([int(f.split('.')[2]) for f in existing_images]) if existing_images else 0
    return count

def augment_image(image):
    augmented_images = []
    
    # Geometric transformations
    rows, cols = image.shape
    # Rotation
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    M2 = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    augmented_images.append(cv2.warpAffine(image, M1, (cols, rows)))
    augmented_images.append(cv2.warpAffine(image, M2, (cols, rows)))
    
    # Flip horizontal
    augmented_images.append(cv2.flip(image, 1))
    
    # Color space transformations
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
    augmented_images.extend([bright, dark])
    
    # Kernel filters
    # Gaussian blur
    blur = cv2.GaussianBlur(image, (5,5), 0)
    augmented_images.append(blur)
    
    # Random erasing
    erased = image.copy()
    x = random.randint(0, rows-20)
    y = random.randint(0, cols-20)
    erased[x:x+20, y:y+20] = 0
    augmented_images.append(erased)
    
    return augmented_images

def collect_frame(name, count, last_capture_time=None):
    global video
    if not video or not video.isOpened():
        return None, count, last_capture_time

    ret, frame = video.read()
    if not ret:
        return None, count, last_capture_time

    current_time = time.time()
    frame_captured = False

    if last_capture_time is None or (current_time - last_capture_time) >= 0.005:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improved preprocessing for better face detection
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Stop collecting if we reach 200 images
            if count >= 200:
                return frame, count, last_capture_time
                
            count += 1
            face_roi = gray[y:y+h, x:x+w]
            
            # Save original image
            original_path = f'datasets/{name}/User.{name}.{count}_original.jpg'
            cv2.imwrite(original_path, face_roi)
            
            # Generate and save augmented images
            augmented_images = augment_image(face_roi)
            for i, aug_img in enumerate(augmented_images):
                aug_path = f'datasets/{name}/User.{name}.{count}_aug_{i+1}.jpg'
                cv2.imwrite(aug_path, aug_img)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
            cv2.putText(frame, str(count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
            last_capture_time = current_time
            frame_captured = True
            break

    return frame, count, last_capture_time

def release_capture():
    global video
    if video:
        video.release()
        video = None

# Feature extraction using HoG and LBPH
def extract_features(image):
    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.train([image], np.array([0]))  # Dummy label for training
    lbph_hist = lbph.getHistograms()[0].flatten()

    hog_features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

    return np.concatenate((lbph_hist, hog_features))

# Train data function
def train_data(name, progress_callback=None):
    features = []
    
    # Load existing models and label map if they exist
    models = {}
    label_map = {}
    if os.path.exists("model/svm_models.pkl"):
        with open("model/svm_models.pkl", "rb") as model_file:
            models = pickle.load(model_file)
    if os.path.exists("model/label_map.pkl"):
        with open("model/label_map.pkl", "rb") as label_map_file:
            label_map = pickle.load(label_map_file)
    
    # Process the selected person's folder
    id_folder_path = os.path.join("datasets", name)
    if not os.path.isdir(id_folder_path):
        raise ValueError(f"No dataset found for {name}")
        
    label_map[name] = len(label_map)  # Assign new ID if not exists
    image_paths = [os.path.join(id_folder_path, f) for f in os.listdir(id_folder_path)]
    total_images = len(image_paths)
    
    if total_images == 0:
        raise ValueError(f"No images found for {name}")
    
    # Extract features
    for idx, image_path in enumerate(image_paths):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (100, 100))
            features.append(extract_features(image))
            
            if progress_callback:
                progress_callback(idx + 1, total_images)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Train model for this person
    svm_model = OneClassSVM(
        kernel='rbf',
        nu=0.005,
        gamma='scale'
    )
    svm_model.fit(features)
    
    # Store model for this person
    models[name] = svm_model
    
    # Save all models and label map
    os.makedirs("model", exist_ok=True)
    with open("model/svm_models.pkl", "wb") as model_file:
        pickle.dump(models, model_file)
    with open("model/label_map.pkl", "wb") as label_map_file:
        pickle.dump(label_map, label_map_file)
        
    print("Training completed.")

def perform_testing_video():
    video = cv2.VideoCapture(0)
    
    # Load all models and label map
    with open("model/svm_models.pkl", "rb") as model_file:
        models = pickle.load(model_file)
    with open("model/label_map.pkl", "rb") as label_map_file:
        label_map = pickle.load(label_map_file)

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face_region, (100, 100))
            features = extract_features(resized_face)
            
            # Test against all models
            max_score = -float('inf')
            recognized_name = "Unknown"
            
            for name, model in models.items():
                decision_score = model.score_samples([features])[0]
                if decision_score > max_score and decision_score > 0.1:  # Threshold check
                    max_score = decision_score
                    recognized_name = name
            
            # Draw results
            if recognized_name != "Unknown":
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{recognized_name} ({max_score:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"Unknown ({max_score:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        yield frame

# ...rest of the code...
