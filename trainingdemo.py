import cv2
import numpy as np
from PIL import Image
import os

recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
sift = cv2.SIFT_create()
path = "datasets"

def train_data(name, progress_callback=None):
    faces = []
    faces_sift = []
    ids = []
    id_to_index = {}
    current_index = 0

    for id_folder in os.listdir(path):
        if id_folder != name:
            continue
        id_folder_path = os.path.join(path, id_folder)
        if not os.path.isdir(id_folder_path):
            continue
        imagePath = [os.path.join(id_folder_path, f) for f in os.listdir(id_folder_path)]
        total_images = len(imagePath)
        
        for idx, imagePaths in enumerate(imagePath):
            faceImage = Image.open(imagePaths).convert('L')
            faceNP = np.array(faceImage)
            keypoints, descriptors = sift.detectAndCompute(faceNP, None)
            str_id = id_folder
            if str_id not in id_to_index:
                id_to_index[str_id] = current_index
                current_index += 1
            numeric_id = id_to_index[str_id]
            if descriptors is not None:
                fixed_size = 100 * 128
                if descriptors.size > 0:
                    if descriptors.shape[0] > 100:
                        descriptors = descriptors[:100]
                    else:
                        pad_size = 100 - descriptors.shape[0]
                        descriptors = np.vstack([descriptors, np.zeros((pad_size, 128))])
                    faces_sift.append(descriptors.flatten())
                else:
                    faces_sift.append(np.zeros(fixed_size))
            faces.append(faceNP)
            ids.append(numeric_id)
            if progress_callback:
                progress_callback(idx + 1, total_images)
    np.save("model/id_mapping.npy", id_to_index)
    recognizer_lbph.train(faces, np.array(ids))
    recognizer_lbph.write("model/Trainer_LBPH.yml")
    np.save("model/sift_features.npy", np.array(faces_sift))
    np.save("model/sift_ids.npy", np.array(ids))
    print("Training completed.")