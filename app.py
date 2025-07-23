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
from threading import Thread
import cv2
from PIL import Image

# Suppress Streamlit warning
class StreamlitWarningFilter(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(StreamlitWarningFilter())

# Constants
MODEL_PATH = "./insightface_model"
EMBEDDINGS_PATH = "embeddings.pkl"
LOG_DIR = "./Recognized_Details"

# Page setup
st.set_page_config(page_title="Face Recognition System", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 6px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
        footer {visibility: hidden;}
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f9fbfd;
            color: #003366;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Header with logo
title_col, logo_col = st.columns([4, 1])
with title_col:
    st.markdown("""
    <h2 style='font-size: 50px; color: #003366; font-weight: bold;'>
        Real-Time Face Recognition System
    </h2>
""", unsafe_allow_html=True)

with logo_col:
    logo_path = "TPL_logo.jpeg"
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        st.image(logo_image, width=120)
    else:
        st.error("Logo file not found. Please check 'TPL_logo.jpeg' path.")

# Session state initialization
if "recognition_process" not in st.session_state:
    st.session_state.recognition_process = None
if "recognition_start_time" not in st.session_state:
    st.session_state.recognition_start_time = None
if "recognition_delay" not in st.session_state:
    st.session_state.recognition_delay = None

# Sidebar: Recognition Controls
st.sidebar.header("Recognition Control")
ip_address = st.sidebar.text_input("Enter Camera Stream URL", value="")
col1, col2 = st.sidebar.columns(2)
start_button = col1.button("Start Recognition")
stop_button = col2.button("Stop Recognition")

# Start recognition
if start_button:
    if not st.session_state.recognition_process or not st.session_state.recognition_process[0].is_alive():
        start_time = time.time()
        recognizer = FaceRecognizer(MODEL_PATH, EMBEDDINGS_PATH, log_dir=LOG_DIR)
        stop_event = threading.Event()
        process = Thread(target=recognizer.recognition_worker, args=(ip_address, stop_event), daemon=True)

        st.session_state.recognition_process = (process, stop_event)
        st.session_state.recognition_start_time = time.time()
        st.session_state.recognition_delay = st.session_state.recognition_start_time - start_time

        process.start()
        st.sidebar.success("")
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
        # st.sidebar.warning("Recognition stopped.")
    else:
        st.sidebar.info("Recognition is not running.")

# Status display
st.sidebar.markdown("---")
st.sidebar.header("Recognition Status")
process_data = st.session_state.recognition_process
if process_data and process_data[0].is_alive():
    delay = st.session_state.get("recognition_delay", None)
    if delay is not None:
        st.sidebar.success(f"Recognition is running. Started after {delay:.2f} seconds.")
    else:
        st.sidebar.success("Recognition is running.")
else:
    st.sidebar.info("Recognition is not running.")

# View Records
st.sidebar.markdown("---")
st.sidebar.header("View Records Details")

def extract_datetime_from_folder(folder_name):
    match = re.search(r"Records_(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})", folder_name)
    if match:
        day, month, year, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute)
    return datetime.min

all_folders = glob.glob(os.path.join(LOG_DIR, "Records_*"))
sorted_folders = sorted(all_folders, key=lambda f: extract_datetime_from_folder(os.path.basename(f)), reverse=True)

if sorted_folders:
    folder_display_names = [os.path.normpath(f).replace("./", "").replace(".\\", "") for f in sorted_folders]
    folder_dict = dict(zip(folder_display_names, sorted_folders))

    selected_display_name = st.sidebar.selectbox("Select Folder", folder_display_names)

    if selected_display_name:
        selected_folder = folder_dict.get(selected_display_name)

        # Show CSV
        csv_files = sorted(glob.glob(os.path.join(selected_folder, "*.csv")), reverse=True)
        if csv_files:
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)

            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(r"^\./|^\.\\", "", regex=True)

            st.subheader("Recognized Details")
            st.dataframe(df)

            st.download_button(
                label="⬇ Download CSV",
                data=df.to_csv(index=False),
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )
        else:
            st.info("No CSV file found in selected folder.")

        # Show recognized images
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
    st.sidebar.info("No folders found.")

# Footer
st.markdown('<div class="footer">Developed by TPL Team</div>', unsafe_allow_html=True)
