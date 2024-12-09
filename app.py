import streamlit as st
import cv2
import numpy as np
import os
# import package form datacollection_train_test.ipynb

from trainingdemo import train_data
from testmodel import perform_testing
from datacollect import initialize_capture, collect_frame, release_capture

# stop_button = st.button("Stop Camera")
# Load models and mappings
recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
# Streamlit app
st.title("Face Recognition App")

menu = ["Collect Data", "Train Data", "Perform Testing"]
choice = st.sidebar.selectbox("Menu", menu)


stop_camera = st.button("Stop Camera")

if choice == "Collect Data":
    st.subheader("Collect Data")
    name = st.text_input("Enter Your Name (no spaces):")
    if name:
        existing_images = len([f for f in os.listdir(os.path.join("datasets", name)) 
                             if f.startswith(f'User.{name}.')]) if os.path.exists(os.path.join("datasets", name)) else 0
        st.info(f"Current number of images for {name}: {existing_images}")
    
    if st.button("Start Collecting", key='start_collect'):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        frame_placeholder = st.empty()
        current_count = initialize_capture(name)
        session_count = 0  # Track images collected in this session
        target_session = 500  # Target for this collection session
        last_capture_time = None
        
        try:
            while not stop_camera and session_count < target_session:
                frame, count, last_capture_time = collect_frame(name, current_count, last_capture_time)
                if frame is not None:
                    frame_placeholder.image(frame, channels="BGR")
                    if count > current_count:  # New image was captured
                        session_count += 1
                    current_count = count
                    progress = session_count / target_session
                    progress_bar.progress(progress)
                    progress_text.text(f'Session progress: {session_count}/{target_session} images (Total: {current_count})')
        finally:
            release_capture()
            
        st.success(f"Collection session completed. Added {session_count} new images. Total images for {name}: {current_count}")

elif choice == "Train Data":
    st.subheader("Train Data")
    dataset_names = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
    name = st.selectbox("Select Name ID to Train:", dataset_names)
    if st.button("Start Training", key='start_train'):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        image_placeholder = st.empty()  # Add image placeholder
        current_image_text = st.empty()  # Add text placeholder for image name
        total_images = len([f for f in os.listdir(os.path.join("datasets", name)) 
                          if f.startswith(f'User.{name}.')])
        
        def progress_callback(current, total):
            progress_bar.progress(current / total)
            progress_text.text(f'Processing image {current} of {total}')
            
        train_data(name, progress_callback)
        st.success(f"Training for {name} completed.")

elif choice == "Perform Testing":
    st.subheader("Perform Testing")
    model_choice = st.selectbox("Choose Model", ["LBPH", "SIFT"])
    if st.button("Start Testing", key='start_test'):
        frame_placeholder = st.empty()
        for frame in perform_testing(model_choice):
            if stop_camera:
                break
            frame_placeholder.image(frame, channels="BGR")
        st.success(f"Testing with {model_choice} model completed.")