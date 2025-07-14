import streamlit as st
import os
import pandas as pd
from new_api_response2 import FaceRecognizer
import glob
from datetime import datetime
import time
import logging
import re
import threading
from threading import Thread, Event

class StreamlitWarningFilter(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()

streamlit_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
streamlit_logger.addFilter(StreamlitWarningFilter())

MODEL_PATH = "./insightface_model"
EMBEDDINGS_PATH = "embeddings.pkl"
LOG_DIR = "./Attendance_Details"

st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("Real-Time Face Recognition Attendance System")

# Session state initialization
if "recognition_process" not in st.session_state:
    st.session_state.recognition_process = None
if "recognition_start_time" not in st.session_state:
    st.session_state.recognition_start_time = None
if "recognition_delay" not in st.session_state:
    st.session_state.recognition_delay = None

# Sidebar controls
st.sidebar.header("Recognition Control")
ip_address = st.sidebar.text_input("Enter IP Camera URL or 0 for Webcam", value="0")

col1, col2 = st.sidebar.columns(2)
start_button = col1.button("Start Recognition")
stop_button = col2.button("Stop Recognition")

# Start recognition
if start_button:
    process_data = st.session_state.recognition_process
    if not process_data or not process_data[0].is_alive():
        start_time = time.time()

        recognizer = FaceRecognizer(MODEL_PATH, EMBEDDINGS_PATH, log_dir=LOG_DIR)
        stop_event = threading.Event()
        process = Thread(target=recognizer.recognition_worker, args=(ip_address, stop_event), daemon=True)

        st.session_state.recognition_process = (process, stop_event)
        st.session_state.recognition_start_time = time.time()
        st.session_state.recognition_delay = st.session_state.recognition_start_time - start_time

        process.start()
    else:
        st.sidebar.info("Recognition is already running.")

# Stop recognition
if stop_button:
    if st.session_state.recognition_process:
        process, stop_event = st.session_state.recognition_process
        stop_event.set()
        st.session_state.recognition_process = None
        st.session_state.recognition_start_time = None
        st.session_state.recognition_delay = None
    else:
        st.sidebar.info("Recognition is not running.")

# Recognition status
st.sidebar.markdown("---")
st.sidebar.header("Recognition Status")
process_data = st.session_state.recognition_process
if process_data and process_data[0].is_alive():
    delay = st.session_state.get("recognition_delay", None)
    if delay is not None:
        st.sidebar.success(f"Recognition is running.\nStarted after {delay:.2f} seconds.")
    else:
        st.sidebar.success("Recognition is running.")
else:
    st.sidebar.info("Recognition is not running.")

# View Attendance Records
st.sidebar.markdown("---")
st.sidebar.header("View Attendance Records")

def extract_datetime_from_folder(folder_name):
    match = re.search(r"Attendance_(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})", folder_name)
    if match:
        day, month, year, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute)
    return datetime.min

all_folders = glob.glob(os.path.join(LOG_DIR, "Attendance_*"))
sorted_folders = sorted(all_folders, key=lambda f: extract_datetime_from_folder(os.path.basename(f)), reverse=True)

if sorted_folders:
    folder_display_names = [os.path.normpath(f).replace("./", "").replace(".\\", "") for f in sorted_folders]
    folder_dict = dict(zip(folder_display_names, sorted_folders))

    selected_display_name = st.sidebar.selectbox("Select Folder", folder_display_names)

    if selected_display_name:
        selected_folder = folder_dict.get(selected_display_name)

        # Load and clean CSV
        csv_files = sorted(glob.glob(os.path.join(selected_folder, "*.csv")), reverse=True)
        if csv_files:
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)

            # Clean './' or '.\' from string columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(r"^\./|^\.\\", "", regex=True)

            st.subheader("Attendance Details")
            st.dataframe(df)

            st.download_button(
                label="⬇ Download CSV",
                data=df.to_csv(index=False),
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )
        else:
            st.info("No CSV file found in selected folder.")

       

        st.subheader("Recognized Images")
        image_dir = os.path.join(selected_folder, "Recognized_images")
        if os.path.exists(image_dir):
            images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            if images:
                cols = st.columns(4)
                for i, img_path in enumerate(images):
                    with cols[i % 4]:
                        img_name = os.path.basename(img_path)

                        
                        match = re.match(r"([a-zA-Z0-9]+)_(\d{2}-\d{2}-\d{4})_(\d{2})-(\d{2})-(\d{2})\.jpg", img_name)
                        if match:
                            name, date_part, hh, mm, ss = match.groups()
                            timestamp = f"{date_part} {hh}:{mm}:{ss}"
                            caption = f"{name} - {timestamp}"
                        else:
                            caption = img_name 

                        st.image(img_path, caption=caption, use_container_width=True)
            else:
                st.info("No recognized images found.")
        else:
            st.warning("Recognized_images folder not found.")
    else:
        st.sidebar.info("No folder selected.")
else:
    st.sidebar.info("No attendance folders found.")
