"""
=============================================================================
YOLOv8 Real-Time Webcam & Phone IP Camera Object Detection System
=============================================================================
Author      : AI & Computer Vision Portfolio
Tech Stack  : Python 3.11, YOLOv8 (Ultralytics), OpenCV, NumPy
Features    : Real-Time Webcam & Phone IP Camera Stream, FPS Counter,
              Target Class Filtering, Screenshot ('s'), Recording ('r')
=============================================================================
"""

import argparse
import os
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# Defined Target Classes as per project specifications
TARGET_CLASSES = [
    "person", "car", "truck", "bottle", "chair", 
    "laptop", "cell phone", "bus", "motorcycle"
]

# Color palette (BGR) for dynamic class drawing
CLASS_COLORS = {
    "person": (255, 99, 71),       # Tomato Red
    "car": (50, 205, 50),          # Lime Green
    "truck": (30, 144, 255),       # Dodger Blue
    "bottle": (255, 215, 0),       # Gold
    "chair": (147, 112, 219),      # Medium Purple
    "laptop": (0, 255, 255),       # Cyan
    "cell phone": (255, 105, 180), # Hot Pink
    "bus": (255, 140, 0),          # Dark Orange
    "motorcycle": (173, 255, 47)   # Green Yellow
}
DEFAULT_COLOR = (0, 255, 127)      # Spring Green fallback

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Webcam & Phone Camera Detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g., 0) OR Phone IP Camera Stream URL (e.g., http://192.168.1.5:8080/video)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--filter-target", action="store_true", help="Only detect the 9 specified target classes")
    return parser.parse_args()

def draw_hud(frame, fps, total_objects, is_recording):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, "YOLOv8 Real-Time AI Vision (Live Dashcam)", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    
    fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"Detected: {total_objects}", (w - 320, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

    if is_recording:
        cv2.circle(frame, (w - 340, 28), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 390, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

def draw_control_panel(frame):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    controls_text = "Hotkeys:  [S] Screenshot  |  [R] Toggle Recording  |  [Q] Quit"
    cv2.putText(frame, controls_text, (20, h - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

def main():
    args = parse_args()
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Determine camera source (Numeric index for local webcam OR string URL for IP Phone Camera)
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("🚀 Initializing YOLOv8 Live Dashcam System")
    print(f"📦 Model           : {args.model}")
    print(f"📹 Stream Source   : {source}")
    print(f"🎯 Confidence Thresh: {args.conf}")
    print(f"🔍 Class Filter     : {'Enabled (9 Target Classes)' if args.filter_target else 'Disabled (All Classes)'}")
    print("=" * 60)

    print("⏳ Loading YOLOv8 model weights...")
    model = YOLO(args.model)
    print("✅ Model loaded successfully!")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Unable to open camera stream at source '{source}'.")
        print("💡 Tip: Make sure your phone and laptop are on the same Wi-Fi network and the IP Camera URL is correct.")
        return

    # If numeric webcam, request high resolution
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps_smooth = 0.0
    is_recording = False
    out_video = None

    print("\nPress 'q' to exit, 's' for screenshot, 'r' to record video.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream interrupted or frame unreadable. Reconnecting...")
            time.sleep(0.5)
            continue

        current_time = time.time()
        time_diff = current_time - prev_time
        prev_time = current_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        total_objects = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if args.filter_target and class_name not in TARGET_CLASSES:
                continue

            total_objects += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.capitalize()} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        draw_hud(frame, fps_smooth, total_objects, is_recording)
        draw_control_panel(frame)

        if is_recording and out_video is not None:
            out_video.write(frame)

        cv2.imshow("YOLOv8 Live Phone Dashcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shot_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")
            cv2.imwrite(shot_path, frame)
            print(f"📸 Screenshot saved to: {shot_path}")
        elif key == ord('r'):
            if not is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rec_path = os.path.join(output_dir, f"dashcam_recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w, _ = frame.shape
                out_video = cv2.VideoWriter(rec_path, fourcc, 20.0, (w, h))
                is_recording = True
                print(f"🔴 Started recording dashcam feed to: {rec_path}")
            else:
                is_recording = False
                if out_video:
                    out_video.release()
                    out_video = None
                print("⏹️ Stopped dashcam recording.")

    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
