import streamlit as st
import os
import pandas as pd
import glob
from datetime import datetime
import logging
import re

# Filter unnecessary Streamlit warnings
class StreamlitWarningFilter(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()

streamlit_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
streamlit_logger.addFilter(StreamlitWarningFilter())

LOG_DIR = "./Attendance_Details"

st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("Real-Time Face Recognition Attendance Viewer")

# Sidebar – View Attendance Records
st.sidebar.header("View Attendance Records")

def extract_datetime_from_folder(folder_name):
    match = re.search(r"Attendance_(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})", folder_name)
    if match:
        day, month, year, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute)
    return datetime.min

# Find and sort attendance folders
all_folders = glob.glob(os.path.join(LOG_DIR, "Attendance_*"))
sorted_folders = sorted(all_folders, key=lambda f: extract_datetime_from_folder(os.path.basename(f)), reverse=True)

if sorted_folders:
    folder_display_names = [os.path.normpath(f).replace("./", "").replace(".\\", "") for f in sorted_folders]
    folder_dict = dict(zip(folder_display_names, sorted_folders))

    selected_display_name = st.sidebar.selectbox("Select Folder", folder_display_names)

    if selected_display_name:
        selected_folder = folder_dict.get(selected_display_name)

        # Load and show CSV
        csv_files = sorted(glob.glob(os.path.join(selected_folder, "*.csv")), reverse=True)
        if csv_files:
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)

            # Clean path entries
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
    st.sidebar.info("No attendance folders found.")
