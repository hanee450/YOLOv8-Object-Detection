"""
=============================================================================
YOLOv8 Real-Time Mobile & Web Object Detector
=============================================================================
Author      : AI & Computer Vision Expert
Tech Stack  : Streamlit, YOLOv8 (Ultralytics), OpenCV, NumPy, Web Speech API
Features    : Dual Mode (Python YOLOv8 AI + HTML5 WebRTC Stream), Real-Time
              Voice Speech Announcer, Glassmorphism UI, High Precision
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

# Custom Glassmorphism Dark Theme Styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            color: #f8fafc;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
        }

        .announcer-box {
            background: linear-gradient(90deg, #0284c7 0%, #6366f1 100%);
            color: white;
            border-radius: 14px;
            padding: 18px 24px;
            font-size: 1.4rem;
            font-weight: 700;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
            margin-bottom: 20px;
            line-height: 1.4;
            text-align: center;
        }

        .main-title {
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.2rem;
        }
        
        .sub-title {
            color: #94a3b8;
            font-size: 1.1rem;
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
            padding: 0.7rem 1.4rem;
            transition: all 0.3s ease;
            width: 100%;
            font-size: 1.1rem;
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
    return YOLO(model_name)

def speak_text_mobile(text: str):
    """Triggers browser Text-to-Speech audio announcement on Android & iPhone speakers."""
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

def annotate_image(image_np, results, conf_threshold, selected_classes, show_labels=True, show_conf=True):
    """Draws high-contrast bounding boxes and labels on an image array."""
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

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

        label_text = class_name.capitalize()
        if show_conf:
            label_text += f" {conf * 100:.1f}%"

        if show_labels:
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 12, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 6, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        class_tallies[class_name] = class_tallies.get(class_name, 0) + 1
        detections.append({
            "Class": class_name.capitalize(),
            "Confidence": f"{conf * 100:.1f}%",
            "BBox": f"({x1}, {y1}, {x2}, {y2})"
        })

    return annotated, detections, class_tallies

# -----------------------------------------------------------------------------
# 4. Main Application Layout & Sidebar Controls
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🚀 YOLOv8 AI Real-Time Object Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Powered by Ultralytics YOLOv8, OpenCV, and Voice Announcer Engine</div>', unsafe_allow_html=True)

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
    enable_voice = st.sidebar.checkbox("🔊 Speaker Voice Announcer", value=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Python YOLOv8 Live Camera", 
        "⚡ HTML5 WebRTC Stream", 
        "🖼️ Image Detection", 
        "🎥 Video Processing"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Python YOLOv8 AI Real-Time Camera Scanner
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📷 Python YOLOv8 AI Camera Object Detection")
        st.write("Scan objects with your laptop webcam or mobile phone camera for instant YOLOv8 detection & spoken voice announcements!")

        if st.button("🔊 Tap Here First to Enable Voice Speaker"):
            speak_text_mobile("Voice Announcer Active!")
            st.success("🔊 Voice Speaker Enabled!")

        auto_loop = st.checkbox("🟢 Continuous Auto-Detection Loop", value=True)

        camera_image = st.camera_input("Point Camera at Objects", key="yolo_python_camera")

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

            if class_tallies:
                tally_items = [f"{count} {cls.capitalize()}{'s' if count > 1 else ''}" for cls, count in class_tallies.items()]
                announcement_text = "I see " + ", ".join(tally_items)
            else:
                announcement_text = "Scanning for objects..."

            st.markdown(f'<div class="announcer-box">📢 <b>LIVE ANNOUNCEMENT:</b><br>{announcement_text}</div>', unsafe_allow_html=True)

            if enable_voice and class_tallies:
                speak_text_mobile(announcement_text)

            st.image(annotated_rgb, caption="YOLOv8 AI Detection Result", use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Objects Detected", len(detections))
            with c2:
                st.metric("Unique Classes", len(class_tallies))
            with c3:
                st.metric("Inference Speed", f"{proc_ms:.1f} ms")

            if detections:
                df_det = pd.DataFrame(detections)
                st.dataframe(df_det, use_container_width=True)

            if auto_loop:
                time.sleep(0.1)
                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 2: HTML5 WebRTC Camera Stream
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("⚡ HTML5 WebRTC Live Video Stream")
        st.write("Zero-click continuous 30 FPS WebRTC camera stream.")

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
            </style>
        </head>
        <body>
            <div class="btn-controls">
                <button class="btn" onclick="switchCamera()">🔄 Flip Camera</button>
            </div>
            <div id="status_text" style="color: #38bdf8; font-weight: bold; margin-bottom: 8px;">⏳ Initializing AI Engine...</div>
            <div class="announcement-card" id="announcement">📢 Live Scan: Scanning...</div>
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
                let model = null;
                let currentFacingMode = "environment";
                let lastSpoken = 0;

                function speak(text) {
                    if (!('speechSynthesis' in window)) return;
                    window.speechSynthesis.cancel();
                    let msg = new SpeechSynthesisUtterance(text);
                    msg.rate = 1.0;
                    window.speechSynthesis.speak(msg);
                }

                async function setupCamera() {
                    if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); }
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: { facingMode: { ideal: currentFacingMode } }, audio: false
                        });
                        video.srcObject = stream;
                        return new Promise((r) => { video.onloadedmetadata = () => r(video); });
                    } catch (e) {
                        statusText.innerText = "⚠️ Allow Camera Permission in browser.";
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
                        let counts = {};
                        predictions.forEach(p => {
                            if (p.score >= 0.35) {
                                const [x, y, w, h] = p.bbox;
                                counts[p.class] = (counts[p.class] || 0) + 1;
                                ctx.strokeStyle = "#00FFFF";
                                ctx.lineWidth = 4;
                                ctx.strokeRect(x, y, w, h);
                                ctx.fillStyle = "#00FFFF";
                                ctx.fillRect(x, y - 25, ctx.measureText(p.class).width + 40, 25);
                                ctx.fillStyle = "#000000";
                                ctx.font = "bold 16px sans-serif";
                                ctx.fillText(`${p.class.toUpperCase()} (${Math.round(p.score * 100)}%)`, x + 5, y - 6);
                            }
                        });
                        let keys = Object.keys(counts);
                        if (keys.length > 0) {
                            let text = "I see " + keys.map(k => `${counts[k]} ${k}`).join(', ');
                            announcementBox.innerText = `📢 ${text}`;
                            let now = Date.now();
                            if (now - lastSpoken > 3000) { speak(text); lastSpoken = now; }
                        }
                    }
                    requestAnimationFrame(detectLoop);
                }

                async function init() {
                    statusText.innerText = "⏳ Loading AI Engine...";
                    model = await cocoSsd.load();
                    statusText.innerText = "⚡ Camera Connected!";
                    await setupCamera();
                    detectLoop();
                }
                init();
            </script>
        </body>
        </html>
        """
        components.html(mobile_live_html, height=600)

    # -------------------------------------------------------------------------
    # TAB 3: Image Detection Mode
    # -------------------------------------------------------------------------
    with tab3:
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
    # TAB 4: Video File Detection Mode
    # -------------------------------------------------------------------------
    with tab4:
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

if __name__ == "__main__":
    main()
