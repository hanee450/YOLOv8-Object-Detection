"""
=============================================================================
YOLOv8 Real-Time Object Detection - Mobile & Web App
=============================================================================
Author      : AI & Computer Vision Expert
Tech Stack  : Streamlit, YOLOv8 (Ultralytics), OpenCV, NumPy, Web Speech API
Features    : Mobile Responsive, Mobile Camera Live Detection, Real-Time
              Voice Speech Announcer on Smartphone Speakers, Dark UI
=============================================================================
"""

import os
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Dark Mode Styling (Mobile Responsive)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 AI Real-Time Mobile & Web Object Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Dark Theme Styling optimized for Mobile & Desktop
st.markdown("""
    <style>
        /* Global Background & Typography */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            color: #f8fafc;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        /* Glassmorphism Containers */
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
        }

        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Mobile Live Announcement Banner */
        .announcer-box {
            background: linear-gradient(90deg, #0284c7 0%, #6366f1 100%);
            color: white;
            border-radius: 14px;
            padding: 14px 20px;
            font-size: 1.25rem;
            font-weight: 700;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35);
            margin-bottom: 16px;
            line-height: 1.4;
        }

        /* Header Accent */
        .main-title {
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.3rem;
            margin-bottom: 0.2rem;
        }
        
        .sub-title {
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        /* Custom Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Custom Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #0284c7 0%, #6366f1 100%);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Constants & Model Caching
# -----------------------------------------------------------------------------
TARGET_CLASSES = [
    "person", "car", "truck", "bottle", "chair", 
    "laptop", "cell phone", "bus", "motorcycle"
]

CLASS_COLORS = {
    "person": (255, 99, 71),       # Red
    "car": (50, 205, 50),          # Green
    "truck": (30, 144, 255),       # Blue
    "bottle": (255, 215, 0),       # Yellow
    "chair": (147, 112, 219),      # Purple
    "laptop": (0, 255, 255),       # Cyan
    "cell phone": (255, 105, 180), # Pink
    "bus": (255, 140, 0),          # Orange
    "motorcycle": (173, 255, 47)   # Lime
}
DEFAULT_COLOR = (0, 255, 127)

@st.cache_resource
def load_yolo_model(model_name: str):
    """Loads and caches the YOLOv8 model weights."""
    return YOLO(model_name)

def speak_text_mobile(text: str):
    """Triggers mobile browser Text-to-Speech audio announcement on Android & iPhone speakers."""
    clean_text = text.replace('"', '').replace("'", "")
    js_code = f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{clean_text}");
                msg.rate = 1.0;
                msg.pitch = 1.0;
                msg.lang = 'en-US';
                window.speechSynthesis.speak(msg);
            }}
        </script>
    """
    components.html(js_code, height=0, width=0)

# -----------------------------------------------------------------------------
# 3. Helper Drawing Functions
# -----------------------------------------------------------------------------
def annotate_image(image_np, results, conf_threshold, selected_classes, show_labels=True, show_conf=True):
    """Draws bounding boxes and labels on an image array."""
    annotated = image_np.copy()
    detections = []
    class_tallies = {}

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]

        if selected_classes and class_name not in selected_classes:
            continue

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        # Draw Rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Build Label
        label_text = class_name.capitalize()
        if show_conf:
            label_text += f" {conf * 100:.1f}%"

        if show_labels:
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Record tally
        class_tallies[class_name] = class_tallies.get(class_name, 0) + 1
        detections.append({
            "Class": class_name.capitalize(),
            "Confidence": f"{conf * 100:.2f}%",
            "BBox (x1,y1,x2,y2)": f"({x1}, {y1}, {x2}, {y2})"
        })

    return annotated, detections, class_tallies

# -----------------------------------------------------------------------------
# 4. Main Application Layout & Sidebar Controls
# -----------------------------------------------------------------------------
def main():
    # Header Section
    st.markdown('<div class="main-title">📱 YOLOv8 Mobile & Web AI Object Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Mobile Camera Detection with Voice Speech Announcer</div>', unsafe_allow_html=True)

    # Sidebar Options
    st.sidebar.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_container_width=True)
    st.sidebar.header("⚙️ Model & Detection Config")

    # Select Model Variant
    model_choice = st.sidebar.selectbox(
        "Select YOLOv8 Weights",
        ["yolov8n.pt (Fastest - Nano)", "yolov8s.pt (Balanced - Small)", "yolov8m.pt (Accurate - Medium)"],
        index=0
    )
    model_weights = model_choice.split(" ")[0]

    # Load Model
    with st.spinner(f"Loading {model_weights}..."):
        model = load_yolo_model(model_weights)

    # Confidence Threshold Slider
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.40,
        step=0.05,
        help="Filter out weak detections with confidence lower than threshold."
    )

    # Class Filter Options
    filter_mode = st.sidebar.radio(
        "Class Detection Filter",
        ["Target Portfolio 9 Classes", "All 80 COCO Classes", "Custom Multiselect"]
    )

    if filter_mode == "Target Portfolio 9 Classes":
        selected_classes = TARGET_CLASSES
    elif filter_mode == "All 80 COCO Classes":
        selected_classes = list(model.names.values())
    else:
        selected_classes = st.sidebar.multiselect(
            "Select Target Classes",
            options=list(model.names.values()),
            default=TARGET_CLASSES
        )

    # UI Options
    st.sidebar.subheader("🎨 Display & Voice Preferences")
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
    enable_voice = st.sidebar.checkbox("🔊 Mobile Speaker Voice Announcer", value=True)

    # Tabs Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📱 Mobile Live Camera & Voice", 
        "🖼️ Image Detection", 
        "🎥 Video Detection", 
        "📊 Model Architecture"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Mobile Live Camera & Real-Time Voice Announcer (Default Tab)
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📱 Smartphone Camera & Real-Time Voice Speech Announcer")
        st.write("Use your phone camera (Android Chrome or iPhone Safari) for live object detection with spoken voice announcements!")

        auto_toggle = st.toggle("🔴 Enable Continuous Auto-Detection Loop", value=True)

        camera_image = st.camera_input("Open Phone Camera", key="mobile_camera_announcer")

        if camera_image is not None:
            bytes_data = camera_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            start_t = time.time()
            results = model.predict(cv2_img_rgb, conf=conf_threshold, verbose=False)[0]
            proc_ms = (time.time() - start_t) * 1000

            annotated_rgb, detections, class_tallies = annotate_image(
                cv2_img_rgb, results, conf_threshold, selected_classes, show_labels, show_conf
            )

            # Build Live Object Announcement String
            if class_tallies:
                tally_items = [f"{count} {cls.capitalize()}{'s' if count > 1 else ''}" for cls, count in class_tallies.items()]
                announcement_text = "I see " + ", ".join(tally_items)
            else:
                announcement_text = "No target objects detected."

            # Prominent Real-Time Live Announcement Box
            st.markdown(f'<div class="announcer-box">📢 <b>LIVE ANNOUNCEMENT:</b><br>{announcement_text}</div>', unsafe_allow_html=True)

            # Trigger Mobile Phone Speaker Voice Speech if enabled
            if enable_voice and class_tallies:
                speak_text_mobile(announcement_text)

            st.image(annotated_rgb, caption="YOLOv8 Mobile Camera Result", use_container_width=True)

            # Analytics Row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Objects Detected", len(detections))
            with c2:
                st.metric("Unique Classes", len(class_tallies))
            with c3:
                st.metric("Inference Speed", f"{proc_ms:.1f} ms")

            if detections:
                df_det = pd.DataFrame(detections)
                st.dataframe(df_det, use_container_width=True)

            # Auto rerun for continuous stream if toggle is ON
            if auto_toggle:
                time.sleep(0.2)
                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 2: Image Detection Mode
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Upload Image for Object Detection")
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            start_time = time.time()
            results = model.predict(image_np, conf=conf_threshold, verbose=False)[0]
            inference_time = (time.time() - start_time) * 1000

            annotated_np, detections, class_tallies = annotate_image(
                image_np, results, conf_threshold, selected_classes, show_labels, show_conf
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_np, caption="YOLOv8 Detected Objects", use_container_width=True)

            if detections:
                df_detections = pd.DataFrame(detections)
                st.dataframe(df_detections, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Video File Detection Mode
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Batch Video File Processing")
        uploaded_video = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if st.button("🚀 Process Video"):
                progress_bar = st.progress(0)
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(frame_rgb, conf=conf_threshold, verbose=False)[0]

                    annotated_rgb, _, _ = annotate_image(
                        frame_rgb, results, conf_threshold, selected_classes, show_labels, show_conf
                    )

                    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                    out.write(annotated_bgr)

                    pct = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                    progress_bar.progress(pct)

                cap.release()
                out.release()
                st.success("🎉 Video processing completed!")

                with open(output_path, "rb") as vf:
                    st.download_button(
                        label="💾 Download Processed Video",
                        data=vf,
                        file_name="yolov8_processed_video.mp4",
                        mime="video/mp4"
                    )

    # -------------------------------------------------------------------------
    # TAB 4: Architecture & Benchmarks
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("📊 YOLOv8 Model Architecture & Performance")

        benchmark_data = {
            "Model Variant": ["YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)", "YOLOv8l (Large)"],
            "Parameters (M)": [3.2, 11.2, 25.9, 43.7],
            "mAP 50-95 (COCO)": [37.3, 44.9, 50.2, 52.9],
            "Recommended Use Case": [
                "Real-time Mobile / Edge Devices",
                "Balanced Mobile & Desktop Projects",
                "High Precision Video Analytics",
                "Server Batch Processing"
            ]
        }

        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench, use_container_width=True)

if __name__ == "__main__":
    main()
