import streamlit as st
from model_dc_t_t import *
import os

# Create model directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

# Streamlit app
st.title("Face Recognition App with HoG + LBPH and SVM")

menu = ["Collect Data", "Train Data", "Perform Testing"]
choice = st.sidebar.selectbox("Menu", menu)

stop_camera = st.button("Stop Camera")

if choice == "Collect Data":
    st.subheader("Collect Data")
    name = st.text_input("Enter Your Name (no spaces):")
    if name:
        # Create datasets directory if it doesn't exist
        if not os.path.exists("datasets"):
            os.makedirs("datasets")
            
        existing_images = len([f for f in os.listdir(os.path.join("datasets", name)) 
                            if f.startswith(f'User.{name}.')]) if os.path.exists(os.path.join("datasets", name))else 0
        st.info(f"Current number of images for {name}: {existing_images}")

    if st.button("Start Collecting", key='start_collect'):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        frame_placeholder = st.empty()
        current_count = initialize_capture(name)
        session_count = 0
        target_session = 200  
        augmented_per_image = 7
        total_expected = target_session * (1 + augmented_per_image)  # Total including augmented images
        last_capture_time = None

        try:
            while not stop_camera and session_count < target_session:
                frame, count, last_capture_time = collect_frame(name, current_count, last_capture_time)
                if frame is not None:
                    frame_placeholder.image(frame, channels="BGR")
                    if count > current_count:
                        session_count += 1
                    current_count = count
                    # Calculate progress including augmented images
                    progress = (session_count * (1 + augmented_per_image)) / total_expected
                    progress_bar.progress(progress)
                    progress_text.text(f'Session progress: {session_count}/{target_session} original images\n'
                                    f'Total images (with augmentation): {session_count * (1 + augmented_per_image)}/{total_expected}')
        finally:
            release_capture()

        st.success(f"Collection session completed. Added {session_count} original images and "
                  f"{session_count * augmented_per_image} augmented images. "
                  f"Total images for {name}: {session_count * (1 + augmented_per_image)}")

elif choice == "Train Data":
    st.subheader("Train Data")
    if not os.path.exists("datasets"):
        st.error("No datasets directory found. Please collect data first.")
    else:
        dataset_names = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
        if not dataset_names:
            st.error("No datasets found. Please collect data first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.selectbox("Select Person to Train:", dataset_names)
                if st.button("Train Selected Person", key='start_train'):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    try:
                        def progress_callback(current, total):
                            progress_bar.progress(current / total)
                            progress_text.text(f'Processing image {current} of {total}')

                        train_data(name, progress_callback)
                        st.success(f"Training completed successfully for {name}!")
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
            
            with col2:
                if st.button("Train All People", key='train_all'):
                    overall_progress = st.progress(0)
                    person_progress = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        total_people = len(dataset_names)
                        for idx, person_name in enumerate(dataset_names):
                            status_text.text(f"Training {person_name} ({idx+1}/{total_people})")
                            
                            def progress_callback(current, total):
                                person_progress.progress(current / total)
                            
                            train_data(person_name, progress_callback)
                            overall_progress.progress((idx + 1) / total_people)
                            
                        st.success(f"Successfully trained all {total_people} people!")
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

elif choice == "Perform Testing":
    st.subheader("Perform Testing")
    if not os.path.exists("model/svm_models.pkl") or not os.path.exists("model/label_map.pkl"):
        st.error("No trained model found. Please train the model first.")
    else:
        frame_placeholder = st.empty()
        if st.button("Start Testing", key='start_test'):
            try:
                for frame in perform_testing_video():
                    if stop_camera:
                        break
                    frame_placeholder.image(frame, channels="BGR")
            except Exception as e:
                st.error(f"Testing failed: {str(e)}")

