import cv2
import numpy as np

def perform_testing(model_choice):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
    recognizer_lbph.read("model/Trainer_LBPH.yml")
    id_mapping = np.load("model/id_mapping.npy", allow_pickle=True).item()
    index_to_id = {v: k for k, v in id_mapping.items()}
    sift_features = np.load("model/sift_features.npy")
    sift_ids = np.load("model/sift_ids.npy")
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)  # Use L2 norm for SIFT

    def get_name_from_id(id):
        return index_to_id.get(id, "Unknown")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            if model_choice == "LBPH":
                lbph_serial, lbph_conf = recognizer_lbph.predict(face_region)
                lbph_score = 100 - min(lbph_conf, 100)
                if lbph_score > 30:
                    person_name = get_name_from_id(lbph_serial)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Conf: {lbph_score:.1f}%", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif model_choice == "SIFT":
                keypoints, descriptors = sift.detectAndCompute(face_region, None)
                if descriptors is not None:
                    descriptors = descriptors.astype(np.float32)
                    try:
                        matches = bf.knnMatch(descriptors, sift_features, k=2)
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.5 * n.distance:  # Adjusted threshold
                                good_matches.append(m)
                        
                        match_confidence = (len(good_matches) / len(matches)) * 100
                        
                        if len(good_matches) > 5:  # Reduced threshold for matches
                            match_idx = [m.trainIdx for m in good_matches]
                            matched_ids = [sift_ids[idx] for idx in match_idx]
                            matched_id = max(set(matched_ids), key=matched_ids.count)
                            person_name = get_name_from_id(matched_id)
                            
                            # Draw rectangle and text
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(frame, f"Conf: {match_confidence:.1f}%", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"SIFT matching error: {e}")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frame = cv2.resize(frame, (640, 480))
        yield frame
        if cv2.waitKey(1) == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    print("Testing completed.")