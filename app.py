import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Traffic AI", layout="wide")

# ----------------------------
# SESSION STATE
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ----------------------------
# WELCOME PAGE
# ----------------------------
if st.session_state.page == "welcome":
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.2);
            padding:50px;
            border-radius:25px;
            backdrop-filter:blur(15px);
            box-shadow:0 8px 32px rgba(0,0,0,0.2);
            text-align:center;">
            <h1>🚦 Smart Traffic Signal Optimization</h1>
            <h3>AI-Powered Dynamic Traffic Control System</h3>
            <br>
            <p style="font-size:18px;">
            An intelligent solution using YOLOv8 to detect vehicles and optimize traffic signals in real-time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    center1, center2, center3 = st.columns([1,2,1])
    with center2:
        if st.button("🚀 Enter System", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    st.stop()  # Stop here for welcome page

# ----------------------------
# MAIN APPLICATION
# ----------------------------
if st.session_state.page == "main":

    # ----------------------------
    # CUSTOM CSS
    # ----------------------------
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); color: white; }
    h1,h2,h3,h4 { text-align:center; color:white; }
    [data-testid="stMetric"] { background-color:#1c1f26; padding:20px; border-radius:12px; border:1px solid #2e3440; box-shadow:0 4px 12px rgba(0,0,0,0.4);}
    [data-testid="stMetricValue"] { color:black !important; font-weight:600; }  /* <-- Make metric values black */
    section[data-testid="stSidebar"] { background-color:#111827; }
    .stButton>button { background-color:#1F77B4; color:white; border-radius:8px; height:3em; width:100%; }
    </style>
    """, unsafe_allow_html=True)

    # ----------------------------
    # TITLE
    # ----------------------------
    st.markdown("""
    <h1>🚦 Smart Traffic Signal Optimization System</h1>
    <h4>AI-Based Dynamic Traffic Control Using Vehicle Detection</h4>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ----------------------------
    # LOAD YOLO MODEL
    # ----------------------------
    @st.cache_resource
    def load_model():
        return YOLO("yolov8l.pt")
    model = load_model()

    # ----------------------------
    # SIDEBAR CONTROL PANEL
    # ----------------------------
    st.sidebar.title("⚙️ Control Panel")
    option = st.sidebar.radio("Select Input Type", ["Upload Video", "Upload Image"])

    # ----------------------------
    # VEHICLE DETECTION
    # ----------------------------
    def process_frame(frame):
        results = model.predict(source=frame, imgsz=1280, conf=0.25, iou=0.5, classes=[1,2,3,5,7], verbose=False)
        vehicle_count = 0
        vehicle_types = {"car":0,"truck":0,"bus":0,"motorcycle":0,"bicycle":0}
        annotated = frame.copy()
        for result in results:
            annotated = result.plot()
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in vehicle_types:
                    vehicle_types[label] += 1
                    vehicle_count += 1
        return annotated, vehicle_count, vehicle_types

    # ----------------------------
    # TRAFFIC LOGIC
    # ----------------------------
    def traffic_logic(vehicle_count):
        if vehicle_count <= 10: return "LOW",15,"🟢 Smooth Traffic"
        elif vehicle_count <= 25: return "MEDIUM",30,"🟡 Moderate Traffic"
        else: return "HIGH",45,"🔴 Heavy Congestion"

    # ----------------------------
    # VIDEO SECTION
    # ----------------------------
    if option=="Upload Video":
        uploaded_video = st.file_uploader("📤 Upload Traffic Video", type=["mp4"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            st.video(tfile.name)
            with st.spinner("Analyzing traffic..."):
                cap = cv2.VideoCapture(tfile.name)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    annotated_frame, vehicle_count, vehicle_types = process_frame(frame)
                    density, signal_time, color_status = traffic_logic(vehicle_count)
                    col1,col2 = st.columns(2)
                    with col1: st.image(annotated_frame, channels="BGR")
                    with col2:
                        m1,m2,m3 = st.columns(3)
                        m1.metric("Total Vehicles", vehicle_count)  # Value now in black
                        m2.metric("Traffic Density", density)
                        m3.metric("Green Signal Time", f"{signal_time} sec")
                        st.markdown(f"### 🚦 {color_status}")

    # ----------------------------
    # IMAGE SECTION
    # ----------------------------
    elif option=="Upload Image":
        uploaded_image = st.file_uploader("📤 Upload Traffic Image", type=["jpg","jpeg","png"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, channels="BGR")
            with st.spinner("Analyzing traffic..."):
                annotated_image, vehicle_count, vehicle_types = process_frame(image)
                density, signal_time, color_status = traffic_logic(vehicle_count)
                col1,col2 = st.columns(2)
                with col1: st.image(annotated_image, channels="BGR")
                with col2:
                    m1,m2,m3 = st.columns(3)
                    m1.metric("Total Vehicles", vehicle_count)  # Value in black
                    m2.metric("Traffic Density", density)
                    m3.metric("Green Signal Time", f"{signal_time} sec")
                    st.markdown(f"### 🚦 {color_status}")

    # ----------------------------
    # VEHICLE TYPE BAR CHART
    # ----------------------------
    if 'vehicle_count' in locals():
        st.markdown("---")
        st.subheader("📊 Vehicle Type Distribution")
        filtered_vehicle_types = {k:v for k,v in vehicle_types.items() if v>0}
        if filtered_vehicle_types:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.bar(filtered_vehicle_types.keys(), filtered_vehicle_types.values())
            ax.set_ylabel("Count")
            ax.set_title("Vehicle Analysis")
            fig.tight_layout()
            st.pyplot(fig)

    # ----------------------------
    # FOOTER
    # ----------------------------
    st.markdown("""
    <hr>
    <div style='text-align:center; font-size:14px; color:lightgray'>
    Smart Traffic Signal Optimization System <br>
    Powered by YOLOv8 | Developed By TEAM IGNITE SQUAD
    </div>
    """, unsafe_allow_html=True)
