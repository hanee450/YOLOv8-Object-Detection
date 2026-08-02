"""
=============================================================================
YOLOv8 Real-Time Mobile & Web AI Object Detector
=============================================================================
Author      : AI & Computer Vision Expert
Tech Stack  : Streamlit, YOLOv8 (Ultralytics), OpenCV, TensorFlow.js COCO-SSD
Features    : 30 FPS Real-Time Smartphone Live Camera Stream, Zero Button Click,
              Phone Speaker Voice Announcer, Camera Flip (Front/Back)
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
# 1. Page Configuration & Custom CSS Dark Mode Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 AI Real-Time Mobile & Web Object Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            color: #f8fafc;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
        }

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

        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

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
    "person": (255, 99, 71),
    "car": (50, 205, 50),
    "truck": (30, 144, 255),
    "bottle": (255, 215, 0),
    "chair": (147, 112, 219),
    "laptop": (0, 255, 255),
    "cell phone": (255, 105, 180),
    "bus": (255, 140, 0),
    "motorcycle": (173, 255, 47)
}
DEFAULT_COLOR = (0, 255, 127)

@st.cache_resource
def load_yolo_model(model_name: str):
    return YOLO(model_name)

def annotate_image(image_np, results, conf_threshold, selected_classes, show_labels=True, show_conf=True):
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

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label_text = class_name.capitalize()
        if show_conf:
            label_text += f" {conf * 100:.1f}%"

        if show_labels:
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        class_tallies[class_name] = class_tallies.get(class_name, 0) + 1
        detections.append({
            "Class": class_name.capitalize(),
            "Confidence": f"{conf * 100:.2f}%",
            "BBox": f"({x1}, {y1}, {x2}, {y2})"
        })

    return annotated, detections, class_tallies

# -----------------------------------------------------------------------------
# 4. Main Application Layout & Sidebar Controls
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">📱 YOLOv8 Mobile & Web AI Object Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Mobile Camera Detection with Voice Speech Announcer</div>', unsafe_allow_html=True)

    st.sidebar.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_container_width=True)
    st.sidebar.header("⚙️ Model & Detection Config")

    model_choice = st.sidebar.selectbox(
        "Select YOLOv8 Weights",
        ["yolov8n.pt (Fastest - Nano)", "yolov8s.pt (Balanced - Small)", "yolov8m.pt (Accurate - Medium)"],
        index=0
    )
    model_weights = model_choice.split(" ")[0]

    with st.spinner(f"Loading {model_weights}..."):
        model = load_yolo_model(model_weights)

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05
    )

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

    st.sidebar.subheader("🎨 Display & Voice Preferences")
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📱 Real-Time Live Camera & Voice", 
        "🖼️ Image Detection", 
        "🎥 Video Processing", 
        "📊 Model Architecture"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: 30 FPS True Real-Time Smartphone Camera + Voice Announcer
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📱 Real-Time Live Smartphone Camera Scanner")
        st.write("Zero-click continuous 30 FPS camera detection with live bounding boxes and voice speech announcements!")

        # Embed Real-Time TensorFlow.js / HTML5 30 FPS AI Object Detection Canvas
        mobile_live_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
            <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
            <style>
                body { margin: 0; padding: 0; background-color: #0f172a; color: white; font-family: sans-serif; text-align: center; }
                .cam-container { position: relative; width: 100%; max-width: 500px; margin: 0 auto; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 2px solid #38bdf8; }
                video { width: 100%; height: auto; display: block; background: #1e293b; }
                canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
                .announcement-card { background: linear-gradient(90deg, #0284c7 0%, #6366f1 100%); color: white; padding: 14px 18px; border-radius: 12px; font-weight: bold; font-size: 1.2rem; margin: 15px auto; max-width: 500px; box-shadow: 0 8px 20px rgba(99,102,241,0.3); }
                .btn-controls { margin: 12px 0; }
                .btn { background: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; font-weight: bold; border-radius: 8px; cursor: pointer; font-size: 1rem; margin: 4px; }
                .btn-voice { background: #10b981; color: white; }
            </style>
        </head>
        <body>

            <div class="btn-controls">
                <button class="btn btn-voice" onclick="toggleVoice()">🔊 Voice Announcer: <span id="voice_status">ON</span></button>
                <button class="btn" onclick="switchCamera()">🔄 Flip Camera</button>
            </div>

            <div id="status_text" style="color: #38bdf8; font-weight: bold; margin-bottom: 8px;">⏳ Initializing AI Vision Engine & Camera...</div>

            <div class="announcement-card" id="announcement">📢 Live Scan: Scanning for objects...</div>

            <div class="cam-container">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="canvas"></canvas>
            </div>

            <script>
                const video = document.getElementById('webcam');
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                const statusText = document.getElementById('status_text');
                const announcementBox = document.getElementById('announcement');
                const voiceStatusBtn = document.getElementById('voice_status');

                let model = null;
                let currentFacingMode = "environment";
                let voiceEnabled = true;
                let lastSpokenTime = 0;

                function toggleVoice() {
                    voiceEnabled = !voiceEnabled;
                    voiceStatusBtn.innerText = voiceEnabled ? "ON" : "OFF";
                    if (voiceEnabled) speak("Voice Announcer Enabled");
                }

                function speak(text) {
                    if (!voiceEnabled || !('speechSynthesis' in window)) return;
                    window.speechSynthesis.cancel();
                    let msg = new SpeechSynthesisUtterance(text);
                    msg.rate = 1.0;
                    msg.lang = 'en-US';
                    window.speechSynthesis.speak(msg);
                }

                async function setupCamera() {
                    if (video.srcObject) {
                        video.srcObject.getTracks().forEach(track => track.stop());
                    }
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: { facingMode: { ideal: currentFacingMode }, width: { ideal: 640 }, height: { ideal: 480 } },
                            audio: false
                        });
                        video.srcObject = stream;
                        return new Promise((resolve) => {
                            video.onloadedmetadata = () => {
                                resolve(video);
                            };
                        });
                    } catch (err) {
                        statusText.innerText = "⚠️ Camera Permission Required. Please allow camera access in browser.";
                    }
                }

                function switchCamera() {
                    currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
                    setupCamera();
                }

                async function detectLoop() {
                    if (video.readyState === 4) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        
                        const predictions = await model.detect(video);
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        let detectedClasses = [];
                        let counts = {};

                        predictions.forEach(prediction => {
                            if (prediction.score >= 0.35) {
                                const [x, y, width, height] = prediction.bbox;
                                const label = prediction.class;
                                counts[label] = (counts[label] || 0) + 1;

                                // Bounding box
                                ctx.strokeStyle = "#00FFFF";
                                ctx.lineWidth = 4;
                                ctx.strokeRect(x, y, width, height);

                                // Label Background
                                ctx.fillStyle = "#00FFFF";
                                const textWidth = ctx.measureText(label).width;
                                ctx.fillRect(x, y - 25, textWidth + 50, 25);

                                // Text
                                ctx.fillStyle = "#000000";
                                ctx.font = "bold 16px sans-serif";
                                ctx.fillText(`${label.toUpperCase()} (${Math.round(prediction.score * 100)}%)`, x + 5, y - 6);
                            }
                        });

                        // Update Announcement Text
                        let keys = Object.keys(counts);
                        if (keys.length > 0) {
                            let itemsText = keys.map(k => `${counts[k]} ${k}`).join(', ');
                            let text = `I see ${itemsText}`;
                            announcementBox.innerText = `📢 Live Announcement: ${text}`;

                            // Voice Speech Every 3 Seconds
                            let now = Date.now();
                            if (now - lastSpokenTime > 3000) {
                                speak(text);
                                lastSpokenTime = now;
                            }
                        } else {
                            announcementBox.innerText = "📢 Live Announcement: Scanning for objects...";
                        }
                    }
                    requestAnimationFrame(detectLoop);
                }

                async function init() {
                    statusText.innerText = "⏳ Loading AI Detection Model...";
                    model = await cocoSsd.load();
                    statusText.innerText = "⚡ Smartphone Live Camera AI Engine Running!";
                    await setupCamera();
                    detectLoop();
                }

                init();
            </script>
        </body>
        </html>
        """

        components.html(mobile_live_html, height=620)

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
