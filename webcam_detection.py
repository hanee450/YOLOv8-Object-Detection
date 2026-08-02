"""
=============================================================================
YOLOv8 Real-Time Object Detection with Voice Speech Announcer (TTS)
=============================================================================
Author      : AI & Computer Vision Portfolio
Tech Stack  : Python 3.11, YOLOv8 (Ultralytics), OpenCV, NumPy, Windows SAPI5
Features    : Real-Time Detection, Native Asynchronous Voice Speech Announcer,
              FPS HUD, Class Filtering, Screenshot ('s'), Video Recording ('r')
=============================================================================
"""

import argparse
import os
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# Defined Target Classes
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
    "bus": (255, 140, 0),          # Dark Orange
    "motorcycle": (173, 255, 47)   # Lime
}
DEFAULT_COLOR = (0, 255, 127)

# Native Windows SAPI5 Speech Engine Initialization
try:
    import win32com.client
    _speaker = win32com.client.Dispatch("SAPI.SpVoice")
except Exception:
    _speaker = None

def speak_text_async(text: str):
    """Native non-blocking asynchronous voice announcer for Windows."""
    print(f"[Voice Announcer]: {text}")
    global _speaker
    if _speaker is not None:
        try:
            # Flag 1 = SVSFlagsAsync (Native Windows background speech, zero FPS drop)
            _speaker.Speak(text, 1)
            return
        except Exception:
            pass
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Voice Object Detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or IP Camera URL")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--filter-target", action="store_true", help="Only detect 9 target classes")
    parser.add_argument("--speak", action="store_true", default=True, help="Enable real-time voice speech announcements")
    parser.add_argument("--voice_interval", type=float, default=3.0, help="Seconds between voice speech announcements")
    return parser.parse_args()

def draw_hud(frame, fps, total_objects, is_recording, announcement):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # Top HUD Bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, "YOLOv8 Real-Time AI Vision", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"Detected: {total_objects}", (w - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

    # Live Announcement Text Banner
    cv2.putText(frame, f"VOICE HUD: {announcement}", (15, 52), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    if is_recording:
        cv2.circle(frame, (w - 320, 25), 6, (0, 0, 255), -1)

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

    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("YOLOv8 Real-Time Voice Object Detection System")
    print(f"Model           : {args.model}")
    print(f"Source          : {source}")
    print(f"Confidence      : {args.conf}")
    print(f"Voice Speech    : {'Enabled' if args.speak else 'Disabled'}")
    print("=" * 60)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera source {source}")
        return

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps_smooth = 0.0
    is_recording = False
    out_video = None

    last_spoken_time = 0
    last_announced_text = "Ready"

    print("\nPress 'q' to quit, 's' to screenshot, 'r' to record video.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        time_diff = current_time - prev_time
        prev_time = current_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        total_objects = 0
        detected_counts = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if args.filter_target and class_name not in TARGET_CLASSES:
                continue

            total_objects += 1
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.capitalize()} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Build Announcement String
        if detected_counts:
            items = [f"{count} {cls.capitalize()}" for cls, count in detected_counts.items()]
            announcement = "Detected " + ", ".join(items)
        else:
            announcement = "Scanning for objects..."

        last_announced_text = announcement

        # Speak announcement via laptop/PC speakers at set intervals
        if args.speak and detected_counts and (current_time - last_spoken_time > args.voice_interval):
            speak_text_async(announcement)
            last_spoken_time = current_time

        draw_hud(frame, fps_smooth, total_objects, is_recording, last_announced_text)
        draw_control_panel(frame)

        if is_recording and out_video is not None:
            out_video.write(frame)

        cv2.imshow("YOLOv8 Voice Object Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shot_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")
            cv2.imwrite(shot_path, frame)
            print(f"Screenshot saved to: {shot_path}")
        elif key == ord('r'):
            if not is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rec_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w, _ = frame.shape
                out_video = cv2.VideoWriter(rec_path, fourcc, 20.0, (w, h))
                is_recording = True
                print(f"Started recording to: {rec_path}")
            else:
                is_recording = False
                if out_video:
                    out_video.release()
                    out_video = None
                print("Stopped recording.")

    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
