import streamlit as st

st.set_page_config(page_title="Test Face Recognition UI", layout="centered")

st.title("📷 Face Recognition Attendance System (Test UI)")

st.write("### Enter RTSP Stream Details")

# Input fields
username = st.text_input("Camera Username", value="admin")
password = st.text_input("Camera Password", type="password")
ip = st.text_input("Camera IP Address", value="192.168.1.100")
port = st.text_input("Port", value="554")
misc = st.text_input("Misc Params", value="channel=1&subtype=0")

# Button to simulate recognition
if st.button("Start Face Recognition"):
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?{misc}"
    
    st.success("✅ Face recognition started.")
    st.write("**RTSP URL Preview:**")
    st.code(rtsp_url)
