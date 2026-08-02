"""
=============================================================================
YOLOv8 Video File Object Detection System
=============================================================================
Author      : AI & Computer Vision Portfolio
Tech Stack  : Python 3.11, YOLOv8 (Ultralytics), OpenCV, NumPy
Features    : Batch Video Processing, Output Export, Object Analytics Summary,
              Class Filtering & Real-time Playback Option
=============================================================================
"""

import argparse
import os
import time
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
    parser = argparse.ArgumentParser(description="YOLOv8 Video File Object Detection")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output/detected_video.mp4", help="Path to save annotated output video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model weights (e.g., yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--filter-target", action="store_true", help="Only detect the 9 specified target classes")
    parser.add_argument("--show", action="store_true", help="Display output window live while processing video")
    return parser.parse_args()

def process_video(input_path, output_path, model_path, conf_thresh, filter_target, show_window):
    if not os.path.exists(input_path):
        print(f"❌ Error: Input video file '{input_path}' not found.")
        return

    # Create parent output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 60)
    print("🚀 YOLOv8 Video File Object Detection & Analytics")
    print(f"📄 Input File      : {input_path}")
    print(f"💾 Output File     : {output_path}")
    print(f"📦 Model Weights   : {model_path}")
    print(f"🎯 Confidence Thresh: {conf_thresh}")
    print(f"🔍 Class Filter     : {'Target 9 Classes' if filter_target else 'All COCO Classes'}")
    print("=" * 60)

    # Load Model
    print("⏳ Loading YOLOv8 model...")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")

    # Open Video Capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {input_path}")
        return

    # Video Properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 Video Metadata: Resolution={width}x{height} | FPS={fps:.1f} | Total Frames={total_frames}")

    # Video Writer Initialization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    start_processing_time = time.time()

    # Object Detection Tally Dictionary
    class_tallies = {}
    total_detection_instances = 0

    print("\n⏳ Processing video frames... Please wait.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame_start = time.time()

        # Run YOLOv8 prediction
        results = model.predict(frame, conf=conf_thresh, verbose=False)[0]

        frame_objects_count = 0

        # Draw Annotations
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if filter_target and class_name not in TARGET_CLASSES:
                continue

            frame_objects_count += 1
            total_detection_instances += 1
            class_tallies[class_name] = class_tallies.get(class_name, 0) + 1

            # Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label box
            label = f"{class_name.capitalize()} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Compute Processing Speed
        frame_time = time.time() - frame_start
        proc_fps = 1.0 / frame_time if frame_time > 0 else 0.0

        # Progress HUD Overlay on Video
        progress_pct = (frame_index / total_frames * 100) if total_frames > 0 else 0
        overlay_text = f"Frame: {frame_index}/{total_frames} ({progress_pct:.1f}%) | Detected: {frame_objects_count} | Speed: {proc_fps:.1f} FPS"
        
        # Bottom status bar
        cv2.rectangle(frame, (0, height - 35), (width, height), (20, 20, 20), -1)
        cv2.putText(frame, overlay_text, (15, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Write annotated frame to output file
        out.write(frame)

        # Console Progress Bar Every 25 Frames
        if frame_index % 25 == 0 or frame_index == total_frames:
            bar_len = 30
            filled_len = int(bar_len * frame_index // total_frames) if total_frames > 0 else 0
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            print(f"\rProgress: [{bar}] {progress_pct:.1f}% ({frame_index}/{total_frames}) | FPS: {proc_fps:.1f}", end="")

        # Show frame in GUI if option set
        if show_window:
            cv2.imshow("YOLOv8 Video Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Video processing cancelled by user.")
                break

    total_proc_time = time.time() - start_processing_time
    avg_fps = frame_index / total_proc_time if total_proc_time > 0 else 0

    # Clean resources
    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    # Print Final Summary Analytics
    print("\n\n" + "=" * 60)
    print("🎉 VIDEO PROCESSING COMPLETED")
    print("=" * 60)
    print(f"⏱️ Total Execution Time : {total_proc_time:.2f} seconds")
    print(f"⚡ Average Speed       : {avg_fps:.1f} FPS")
    print(f"🔢 Total Frame Count   : {frame_index}")
    print(f"🎯 Total Detections    : {total_detection_instances} objects")
    print("-" * 60)
    print("📊 DETECTION TALLY BY CLASS:")
    if class_tallies:
        for cls, count in sorted(class_tallies.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {cls.capitalize():<15}: {count} detections")
    else:
        print("   (No objects detected matching criteria)")
    print("=" * 60)
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    args = parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        conf_thresh=args.conf,
        filter_target=args.filter_target,
        show_window=args.show
    )
