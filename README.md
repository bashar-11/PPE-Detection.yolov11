# 🦺 PPE Detection with YOLOv11

**Real-time Personal Protective Equipment (PPE) detection system** built on Ultralytics **YOLOv11** and served through an interactive **Streamlit** web application. The system identifies safety compliance (helmets, vests, and related PPE) from images, uploaded videos, and a live webcam feed — helping construction sites, factories, and industrial facilities automate safety monitoring.

**🌐 Live Demo:** [ppe-detection-11.streamlit.app](https://ppe-detection-11.streamlit.app/) &nbsp;|&nbsp;

**🎬 Sample Test Videos:** [Google Drive](https://drive.google.com/file/d/1AKw4ScHoclA-ajLdfqzwskElFCzgbkl3/view?usp=sharing)

---

## 📌 Overview

Manual supervision of PPE compliance on job sites is slow, inconsistent, and hard to scale. **PPE-Detection.yolov11** solves this by pairing a fine-tuned **YOLOv11** object-detection model with a lightweight, no-code **Streamlit dashboard**, so anyone — not just ML engineers — can drop in an image, a video, or a live camera stream and instantly see who is (and isn't) following safety protocol.

**Why this project:**
- 🎯 **Accurate** — trained on a large, purpose-built PPE dataset (10,000+ annotated images).
- ⚡ **Fast** — YOLOv11's single-pass architecture enables real-time inference on video and webcam streams.
- 🖥️ **Accessible** — a polished Streamlit UI means no command-line expertise is needed to run detections.
- 🧩 **Configurable** — confidence, IoU, inference resolution, and frame-skip are all tunable from the sidebar.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image detection** | Upload a single image and get side-by-side original vs. annotated results, plus a detections table with class and confidence. |
| 🎥 **Video detection** | Upload a video and watch synchronized original/annotated playback, with adjustable frame-skip for speed. |
| 🔴 **Live camera detection** | Run real-time PPE detection directly from your webcam. |
| 🎚️ **Tunable inference** | Adjust confidence threshold, IoU threshold, inference resolution (320/480/640), and frame skip on the fly. |
| 📊 **Detection summary** | Per-run object counts and a confidence table for every detected class. |
| 🧠 **Custom-trained YOLOv11 model** | Fine-tuned specifically for PPE / safety-gear detection rather than generic COCO classes. |

---

## 🎬 Demo

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2051/2051515.png" width="90"/>
</p>

### 🌐 Try it live
No installation needed — the app is deployed on Streamlit Community Cloud:

👉 **[https://ppe-detection-11.streamlit.app](https://ppe-detection-11.streamlit.app/)**

Open it in your browser and use the **Image Upload**, **Video Upload**, or **Live Camera** tabs directly.

### 🎥 Sample test videos
Don't have a video on hand to test with? Use these sample clips (construction/safety footage) to try out the **Video Upload** tab:

👉 **[Download test videos (Google Drive)](https://drive.google.com/file/d/1AKw4ScHoclA-ajLdfqzwskElFCzgbkl3/view?usp=sharing)**

1. Download a video from the link above.
2. Open the [live app](https://ppe-detection-11.streamlit.app/) (or run it locally — see [Quick Start](#-quick-start)).
3. Go to the **🎥 Video Upload** tab, upload the file, and click **▶️ Start Synchronized Processing**.

> Consider adding a screenshot or GIF of the app here once available, e.g.:
>
> ```markdown
> ![Demo](docs/demo.gif)
> ```

---

## 🗂️ Project Structure

```
PPE-Detection.yolov11/
├── app.py                     # Streamlit web application (image / video / live camera tabs)
├── requirements.txt           # Python dependencies
├── structure.txt              # Auto-generated project tree snapshot
├── configs/                   # Training / model configuration files (e.g. data.yaml, hyperparameters)
├── data/
│   └── raw/
│       └── ppe_dataset/       # Raw PPE dataset (images + YOLO-format labels)
├── models/
│   └── best.pt                # Trained YOLOv11 weights used by the app
├── src/                       # Source code: training, evaluation, and utility scripts
├── tests/                     # Unit / integration tests
├── .vscode/                   # Editor configuration
├── README.md                  # You are here
├── README.dataset.md          # Dataset attribution (Roboflow export)
└── README.roboflow.md         # Roboflow export details & license
```

> 💡 Exact contents of `configs/`, `src/`, and `tests/` may evolve — check those folders directly for the latest scripts (e.g. training, data preparation, evaluation, and test suites).

---

## 🧠 Model

- **Architecture:** [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/) — the latest generation of the YOLO real-time object detection family.
- **Task:** Object detection (bounding boxes + class labels).
- **Weights file:** `models/best.pt` — the best checkpoint produced during training, loaded automatically by `app.py`.
- **Classes:** PPE / safety-related categories such as helmet, vest, and related compliance classes (see `configs/data.yaml` and `models/` for the exact, current class list and mapping).
- **Framework:** PyTorch, via the `ultralytics` package.

If you retrain the model, drop the new best checkpoint into `models/best.pt` (or update `MODEL_PATH` in `app.py`) to have the app pick it up automatically.

---

## 📦 Dataset

The model is trained on a PPE detection dataset exported from **[Roboflow](https://roboflow.com)**:

- **Source:** [PPE Detection dataset — Roboflow Universe](https://universe.roboflow.com/b-abdelraziq/ppe-detection-vwjor-aqptb)
- **Size:** ~10,090 annotated images
- **Format:** YOLO (YOLOv8/YOLOv11-compatible) bounding-box annotations
- **License:** CC BY 4.0
- **Preprocessing:** No preprocessing or augmentation applied at export time (see `README.dataset.md` / `README.roboflow.md` for the original Roboflow export notes)

Raw data lives under `data/raw/ppe_dataset/`. See [`README.dataset.md`](README.dataset.md) and [`README.roboflow.md`](README.roboflow.md) for full dataset attribution and export details.

---

## ⚙️ Installation

### Prerequisites
- Python **3.9+**
- `pip`
- (Optional but recommended) a virtual environment tool such as `venv` or `conda`
- A CUDA-capable GPU is recommended for training and faster inference, but the app runs on CPU as well

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/bashar-11/PPE-Detection.yolov11.git
cd PPE-Detection.yolov11

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies** (from `requirements.txt`): `streamlit`, `ultralytics`, `opencv-python-headless`, `pillow`, `pandas`, `numpy`, `torch`, `torchvision`.

---

## 🚀 Quick Start

> 💡 **Just want to try it?** Skip the setup and use the hosted app: **[ppe-detection-11.streamlit.app](https://ppe-detection-11.streamlit.app/)** — no installation required.

To run it locally instead: make sure a trained model checkpoint exists at `models/best.pt`, then launch the app:

```bash
streamlit run app.py
```

This opens the **Smart Safety Monitoring System** dashboard in your browser (default: `http://localhost:8501`), where you can:

1. **🖼️ Image Upload** — upload a `.jpg`/`.jpeg`/`.png` file and view detections instantly.
2. **🎥 Video Upload** — upload a `.mp4`/`.avi`/`.mov` file and click **Start Synchronized Processing** to watch original vs. annotated playback side by side.
3. **🔴 Live Camera** — click **Start Camera** to run detection on your webcam feed in real time, and **Stop Camera** to end the session.

Use the sidebar to tune:
- **🎯 Confidence Threshold** — minimum detection confidence to display a result.
- **📏 IoU Threshold** — non-max suppression overlap threshold.
- **📐 Inference Resolution** — 320 / 480 / 640 px; lower values speed up video/webcam processing.
- **⏩ Frame Skip** — process every N-th frame for faster video playback.

---

## 🏋️ Training / Fine-tuning

To train or fine-tune YOLOv11 on this (or your own) PPE dataset using the Ultralytics API:

```python
from ultralytics import YOLO

# Start from a pretrained YOLOv11 checkpoint
model = YOLO("yolo11n.pt")   # or yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt

# Train on the PPE dataset (update path to your data.yaml)
model.train(
    data="configs/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

After training, copy the resulting best checkpoint (typically `runs/detect/train/weights/best.pt`) into `models/best.pt` so the Streamlit app can use it.

> See the `src/` and `configs/` folders for any project-specific training, data-preparation, or evaluation scripts already included in this repository.

---

## 🧪 Testing

If test scripts are provided under `tests/`, run them with:

```bash
pytest tests/
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Detection model | Ultralytics YOLOv11 (PyTorch) |
| Web app / UI | Streamlit |
| Image & video processing | OpenCV, Pillow |
| Data handling | Pandas, NumPy |
| Dataset tooling | Roboflow |

---

## 🗺️ Roadmap Ideas

- [ ] Add downloadable annotated output (image/video export)
- [ ] Add per-class alerting (e.g. flag "no-helmet"/"no-vest" detections)
- [ ] Add model performance report (mAP, precision, recall) to the README
- [ ] Dockerize the app for one-command deployment
- [ ] Add RTSP/IP camera stream support alongside local webcam

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please open an issue first for major changes so we can discuss what you'd like to change.

---

## 📄 License

- **Code:** No explicit code license has been specified for this repository yet. Consider adding one (e.g. MIT, Apache-2.0) via a `LICENSE` file — until then, all rights are reserved by the author by default.
- **Dataset:** Licensed **CC BY 4.0**, courtesy of the original Roboflow contributor. See [`README.dataset.md`](README.dataset.md) and [`README.roboflow.md`](README.roboflow.md) for full attribution.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 model and training framework
- [Roboflow](https://roboflow.com) for dataset hosting, annotation, and export tooling
- [Streamlit](https://streamlit.io) for the interactive web app framework
- The Roboflow community contributor behind the [PPE Detection dataset](https://universe.roboflow.com/b-abdelraziq/ppe-detection-vwjor-aqptb)

---

## 📬 Contact

Maintained by **[bashar-11](https://github.com/bashar-11)**. For questions, issues, or feature requests, please [open an issue](https://github.com/bashar-11/PPE-Detection.yolov11/issues) on GitHub.
