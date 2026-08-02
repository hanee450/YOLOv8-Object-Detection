# 🚀 Real-Time Object Detection System using YOLOv8, OpenCV & Streamlit

![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A production-ready **Real-Time Object Detection & Analytics System** built using **YOLOv8**, **OpenCV**, and **Streamlit**. Designed as a high-impact portfolio project for AI Engineers, Data Scientists, and Computer Vision specialists.

---

## 🌟 Key Features

* 📹 **Real-Time Webcam Detection**: Low-latency video stream processing with moving average FPS calculation.
* 🎥 **Batch Video Processing**: Process pre-recorded videos (`.mp4`, `.avi`, `.mov`) with frame progress metrics and annotated video export.
* 🌐 **Interactive Streamlit Web App**: Dark-mode Glassmorphism dashboard featuring file uploads, confidence threshold sliders, dynamic target class filters, and export capabilities.
* 🎯 **9 Core Target Classes**: Out-of-the-box filtering for `person`, `car`, `truck`, `bus`, `motorcycle`, `bottle`, `chair`, `laptop`, and `cell phone`.
* 📸 **Instant Hotkey Controls**: Capture screenshots (`'s'`) or record live webcam feed (`'r'`) on demand.
* 📊 **Analytical Visualizations**: Interactive object frequency distribution charts and tabular breakdown of bounding box coordinates.

---

## 📁 Project Structure

```text
YOLOv8-Object-Detection/
│── app.py                 # Interactive Streamlit Web Dashboard (Dark Theme & Analytics)
│── webcam_detection.py    # Standalone real-time webcam detector script with hotkeys
│── video_detection.py     # Standalone batch video processing script
│── requirements.txt       # Project dependencies with pinned compatible versions
│── README.md              # Documentation & guide
│── LINKEDIN_POST.md       # Recruiter-focused LinkedIn post template
│── assets/                # Screenshots and project assets
│   └── .gitkeep
└── output/                # Output directory for screenshots & processed videos
    └── .gitkeep
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
Ensure you have **Python 3.11** (or Python 3.9+) installed on your machine.

### 2. Clone Repository
```bash
git clone https://github.com/your-username/YOLOv8-Object-Detection.git
cd YOLOv8-Object-Detection
```

### 3. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Alternatively, install packages directly)*:
```bash
pip install ultralytics opencv-python numpy streamlit pillow pandas
```

---

## 💻 Usage

### 1. Run Interactive Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
> Open your browser at `http://localhost:8501` to access the dark-theme detection interface.

---

### 2. Run Real-Time Webcam Detection
```bash
# Default run (All COCO classes, webcam index 0, confidence 0.45)
python webcam_detection.py

# Filter only the 9 target portfolio classes
python webcam_detection.py --filter-target --conf 0.50
```
**Webcam Hotkeys:**
* Press `s` — Save instant screenshot to `output/`
* Press `r` — Start / Stop video recording
* Press `q` — Quit webcam stream

---

### 3. Run Batch Video File Detection
```bash
# Process input video file and save output video
python video_detection.py --input sample_video.mp4 --output output/result_video.mp4 --conf 0.40 --filter-target
```

---

## 📊 Performance Benchmarks (COCO Dataset)

| Model Variant | Parameters | FLOPs | mAP 50-95 | CPU Inference | Recommended Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n (Nano)** | 3.2M | 8.7B | 37.3 | ~80 ms | Real-Time Webcam & Edge Devices |
| **YOLOv8s (Small)** | 11.2M | 28.6B | 44.9 | ~128 ms | Desktop Real-Time Analytics |
| **YOLOv8m (Medium)**| 25.9M | 78.9B | 50.2 | ~235 ms | High Precision Server Processing |

---

## ☁️ Deploying on Streamlit Cloud

1. Push your repository to **GitHub**.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select your repository, set main file to `app.py`.
4. Click **Deploy!**

---

## 🔮 Future Improvements

* [ ] Add DeepSORT / ByteTRACK for Multi-Object Tracking (MOT) across frames.
* [ ] Train custom YOLOv8 model on custom domain-specific dataset (e.g., Defect Detection).
* [ ] Convert model to TensorRT / OpenVINO for sub-10ms inference.

---

## 👤 Author & Contact

Developed by **AI & Computer Vision Portfolio**  
* 💼 **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)  
* 🐙 **GitHub**: [Your GitHub Profile](https://github.com/your-username)  

---

*⭐ If you found this project helpful, please give it a star on GitHub!*
