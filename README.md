<div align="center">

# PPE Detection with YOLOv11

**Real-time Personal Protective Equipment compliance detection**, fine-tuned on Ultralytics YOLOv11 and served through an interactive Streamlit dashboard.

[![CI](https://github.com/bashar-11/PPE-Detection.yolov11/actions/workflows/ci.yml/badge.svg)](https://github.com/bashar-11/PPE-Detection.yolov11/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-FF0000?logo=ultralytics)](https://docs.ultralytics.com/models/yolo11/)

[Live demo](https://ppe-detection-11.streamlit.app) · [Dataset & attribution](README.dataset.md) · [Report a bug](.github/ISSUE_TEMPLATE/bug_report.md)

</div>

---

## What it does

Upload an image, a video, or point a webcam at a worker, and the model marks
every **safety helmet** and **reflective jacket** it can see. The point is not
the detection — it is turning detection into something a site supervisor can
act on without knowing how to run a Python script.

## Classes

This model detects exactly **two** classes, matching the training dataset:

| ID | Class |
|---|---|
| 0 | `Reflective-Jacket` |
| 1 | `Safety-Helmet` |

Gloves, masks, eye protection and footwear are **not** part of this model. If
you need them, train on a dataset that labels them — see [Extending](#extending).

## Quick start

```bash
git clone https://github.com/bashar-11/PPE-Detection.yolov11.git
cd PPE-Detection.yolov11

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The trained weights (`models/best.pt`, ~19 MB) are committed, so the app runs
straight away — no training, no dataset download, no API key.

## The dashboard

| Tab | What it does |
|---|---|
| **Image Upload** | Side-by-side original and annotated image, plus a table of detected classes and confidences. |
| **Video Upload** | Synchronised original and annotated playback. Adjustable frame skip for speed. |
| **Live Camera** | Real-time detection on your webcam, with an on-screen FPS counter. |

The sidebar exposes confidence, IoU, inference resolution (320/480/640) and
frame skip. Resolution and frame skip are the two that matter for video: the
model still runs comfortably on CPU, but lowering the resolution or skipping
frames keeps playback smooth.

## Model performance

Measured on the held-out **test split** with `conf=0.25`, `IoU=0.70`,
`imgsz=640`.

| Metric | Value |
|---|---|
| Precision | — |
| Recall | — |
| mAP@50 | — |
| mAP@50-95 | — |

> **These numbers are generated, not typed.** Run the evaluation and paste the
> output here so the table can never drift from the weights in the repo:
>
> ```bash
> python -m src.evaluate
> ```
>
> It prints the four values above and writes `runs/evaluation/metrics.json`.

## Retraining

The dataset is not in this repository — it is redistributed by Roboflow under
CC BY 4.0. To retrain or re-evaluate you need it on disk:

1. Get the dataset from the source in [README.dataset.md](README.dataset.md).
2. Extract it so that `data/raw/ppe_dataset/` contains `train/`, `valid/` and
   `test/`. The class definition already lives in `configs/data.yaml`.
3. Inspect it:

   ```bash
   python -m src.analysis.dataset_analyzer
   ```

4. Train:

   ```bash
   python -m src.training.train
   ```

5. Evaluate, then copy the best checkpoint over the committed one:

   ```bash
   python -m src.evaluate
   cp runs/train/ppe_yolo11s/weights/best.pt models/best.pt
   ```

Training settings live in [`configs/train_config.yaml`](configs/train_config.yaml);
inference defaults in [`configs/inference_config.yaml`](configs/inference_config.yaml).
The current checkpoint was trained for 100 epochs from `yolo11s.pt` at 640px
with `batch: -1` (auto-sized) and early stopping at 20 epochs of no improvement.

## Script reference

| Command | Purpose |
|---|---|
| `python -m src.training.train` | Fine-tune from a YOLO11 checkpoint. |
| `python -m src.evaluate` | Report precision/recall/mAP on the test split. |
| `python -m src.predict` | Batch detection over a folder of images. |
| `python -m src.predict_video --source clip.mp4` | Detection over a video file. |
| `python -m src.webcam` | Live detection from a webcam. |
| `python -m src.analysis.dataset_analyzer` | Dataset class list and per-split counts. |

Every script reads its defaults from `configs/` and accepts `--conf`, `--imgsz`
and `--model` overrides.

## Project structure

```
PPE-Detection.yolov11/
├── app.py                     # Streamlit dashboard
├── configs/
│   ├── data.yaml              # Dataset definition and the two class names
│   ├── train_config.yaml      # Training hyperparameters
│   └── inference_config.yaml  # Default inference settings
├── src/
│   ├── utils.py               # Config loading, device selection, metrics helpers
│   ├── evaluate.py            # Test-split evaluation, writes metrics.json
│   ├── predict.py             # Batch image inference
│   ├── predict_video.py       # Video inference
│   ├── webcam.py              # Live webcam inference
│   ├── training/train.py      # Fine-tuning entrypoint
│   └── analysis/dataset_analyzer.py
├── models/best.pt             # Trained weights (committed on purpose)
├── tests/                     # pytest suite, no model or dataset needed
├── .github/workflows/ci.yml   # Lint, test, hygiene checks, security scan
└── CONTRIBUTING.md
```

## Design notes

- **Heavy imports live inside functions.** `ultralytics`, `torch` and `cv2` are
  imported where they are used, not at module top level. That is what lets the
  test suite run in seconds without loading a 19 MB model.
- **The dataset is git-ignored, and CI enforces it.** An earlier version of this
  repository carried ~20,000 dataset files, a 48 MB test video and a generated
  tree dump, which made the clone take minutes. The hygiene job in CI now fails
  the build if any of that comes back.
- **Video defaults are stricter than image defaults.** `src/webcam.py` uses
  `conf=0.30` rather than 0.25, because video frames carry sensor noise and
  compression artefacts that otherwise produce detections flickering in and out.

## Extending

- **More classes** — train on a dataset that includes them and update
  `configs/data.yaml`. The class names are the only thing the app needs.
- **RTSP / IP cameras** — `src/webcam.py` uses `cv2.VideoCapture(0)`; an RTSP
  URL works as a drop-in replacement for the index.
- **Violation alerts** — the app already surfaces detections; flagging
  "no helmet in frame" is a short addition on top of the existing results.
- **Export to ONNX/TensorRT** — Ultralytics exports directly; keep the output
  out of git.

## Known limitations

- Two classes only, inherited from the training dataset.
- The webcam tab needs a browser that permits camera access; over plain HTTP on
  a non-localhost host, browsers block it.
- CPU inference is noticeably slower than GPU, especially on video.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11 and the
  training/validation tooling.
- [Roboflow Universe](https://universe.roboflow.com/) and the original dataset
  contributor — the PPE Detection dataset (CC BY 4.0).
- [Streamlit](https://streamlit.io) — the dashboard framework.

## License

**Code and trained weights:** [MIT](LICENSE) © 2026 Bashar Sayed.

**Dataset:** CC BY 4.0, © its original contributor. It is not redistributed
here — see [README.dataset.md](README.dataset.md).