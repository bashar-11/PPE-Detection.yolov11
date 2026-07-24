import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2
from pathlib import Path
import tempfile
import time

# ==========================
# Page Config & Custom CSS
# ==========================
st.set_page_config(
    page_title="Smart Safety Monitoring",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .st-tabs {font-weight: bold;}
    h1 {color: #1f2937; font-family: 'Inter', sans-serif;}
    
    /* منع تمدد وبكسلة الصور وحفظ الأبعاد الأصلية */
    .stImage img {
        max-height: 480px !important;
        width: auto !important;
        object-fit: contain !important;
        margin: 0 auto;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .video-label {
        font-weight: 700;
        margin-bottom: 8px;
        color: #374151;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# ==========================
# Sidebar Settings
# ==========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2051/2051515.png", width=80)
    st.title("⚙️ Model Settings")
    st.markdown("---")
    confidence = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    iou = st.slider("📏 IoU Threshold", 0.1, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.subheader("⚡ Performance Settings")
    img_size = st.select_slider("📐 Inference Resolution", options=[320, 480, 640], value=640, help="Lower resolution increases video processing speed.")
    frame_skip = st.slider("⏩ Frame Skip (Speedup)", 1, 5, 1, help="Process every N-th frame for faster playback.")
    
    st.markdown("---")
    st.info("💡 **Tip:** Reduce Inference Resolution or increase Frame Skip if video playback feels slow.")

# ==========================
# Main Title
# ==========================
st.title("🦺 Smart Safety Monitoring System")
st.markdown("Real-time object detection for safety and monitoring.")
st.markdown("---")

# ==========================
# Tabs Setup
# ==========================
tab1, tab2, tab3 = st.tabs(["🖼️ Image Upload", "🎥 Video Upload", "🔴 Live Camera"])

# --------------------------
# TAB 1: Image Processing
# --------------------------
with tab1:
    st.subheader("Detect Objects in Images")
    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="img_uploader")
    
    if uploaded_img:
        image = Image.open(uploaded_img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='video-label'>📷 Original Image</div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with st.spinner("Processing..."):
            results = model.predict(image, conf=confidence, iou=iou, imgsz=img_size)
            annotated_img = results[0].plot()
            
        with col2:
            st.markdown("<div class='video-label'>🔍 Detection Result</div>", unsafe_allow_html=True)
            st.image(annotated_img, channels="BGR", use_container_width=True)
            
        # Results Table
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            detections.append({
                "Class": model.names[cls].title(),
                "Confidence": f"{conf:.2%}"
            })
            
        if detections:
            st.success(f"✅ Detected {len(detections)} object(s).")
            df = pd.DataFrame(detections)
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Total Objects", len(df))
            with col_b:
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ No Objects Detected")

# --------------------------
# TAB 2: Video Processing
# --------------------------
with tab2:
    st.subheader("Detect Objects in Recorded Videos")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"], key="vid_uploader")
    
    if uploaded_video:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        st.markdown("---")
        start_btn = st.button("▶️ Start Synchronized Processing", type="primary", use_container_width=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='video-label'>📹 Original Stream</div>", unsafe_allow_html=True)
            orig_placeholder = st.empty()
        with col2:
            st.markdown("<div class='video-label'>🎯 Detection Stream</div>", unsafe_allow_html=True)
            proc_placeholder = st.empty()
            
        if start_btn:
            cap = cv2.VideoCapture(tfile.name)
            frame_count = 0
            
            # Temporary storage for last processed annotated frame for skipped frames
            last_annotated_rgb = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert original frame BGR to RGB
                orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame based on frame_skip setting
                if frame_count % frame_skip == 0 or last_annotated_rgb is None:
                    # Run Inference
                    results = model.predict(frame, conf=confidence, iou=iou, imgsz=img_size, verbose=False)
                    annotated_frame = results[0].plot()
                    last_annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display both original and detected frames side by side in PERFECT SYNC and same size
                orig_placeholder.image(orig_rgb, channels="RGB", use_container_width=True)
                proc_placeholder.image(last_annotated_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
            st.success("✅ Synchronized video processing completed!")

# --------------------------
# TAB 3: Live Camera
# --------------------------
with tab3:
    st.subheader("Real-Time Camera Detection")
    st.markdown("Click **Start** to open your webcam and run detections in real-time.")
    
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        start_cam = st.button("🔴 Start Camera", use_container_width=True)
    with col2:
        stop_cam = st.button("⏹️ Stop Camera", use_container_width=True)
        
    cam_placeholder = st.empty()
    
    if start_cam:
        cap = cv2.VideoCapture(0) # 0 is the default local camera
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            st.session_state['run_cam'] = True
            
            while st.session_state.get('run_cam', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Error reading camera frame.")
                    break
                    
                results = model.predict(frame, conf=confidence, iou=iou, imgsz=img_size, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                cam_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
    if stop_cam:
        st.session_state['run_cam'] = False
        cam_placeholder.info("Camera stopped.")