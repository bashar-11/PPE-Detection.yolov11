# Contributing to PPE Detection (YOLOv11)

Thanks for contributing. This document covers setup and the conventions that
keep the repository small enough to clone.

## Setup

```bash
git clone https://github.com/bashar-11/PPE-Detection.yolov11.git
cd PPE-Detection.yolov11

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

The trained weights (`models/best.pt`) are committed, so the app runs
immediately without training.

The dataset is **not** in this repository. See
[README.dataset.md](README.dataset.md) — download it and extract it to
`data/raw/ppe_dataset/` if you want to retrain or re-evaluate.

## Before you open a pull request

```bash
ruff check src tests app.py     # lint
ruff format src tests app.py    # format
pytest tests -q                 # tests
```

CI runs all three, plus a repository hygiene check and a dependency audit.

## Keeping the repository small

This project previously carried the full dataset (over 20,000 files), a test
video and a generated tree dump, which made the clone take minutes. CI now
fails the build if any of that comes back. Concretely:

- **Datasets, videos and sample images** go to a Release asset or an external
  link — never into git.
- **Generated reports** (`structure.txt`) are not committed.
- **Weights**: `models/best.pt` is the one deliberate exception, because the
  app needs it at startup. Any other checkpoint is reproducible; keep it out.
- Nothing over **50 MB** may be committed.

## Publishing metrics

Model numbers should never be typed by hand. Run:

```bash
python -m src.evaluate
```

This prints precision, recall, mAP@50 and mAP@50-95 and writes them to
`runs/evaluation/metrics.json`. Paste those exact values into the README table
when you replace the weights, so the published numbers always match the model
in the repository.

## Adding a feature

Keep the scripts importable: heavy imports (`ultralytics`, `torch`, `cv2`)
belong inside the function that needs them, not at module top level. That is
what lets the test suite run without loading a model.