from pathlib import Path
from ultralytics import YOLO

# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SOURCE_DIR = PROJECT_ROOT / "tests"  / "test_img"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "predict"

# =========================
# Load Model
# =========================

model = YOLO(str(MODEL_PATH))

# =========================
# Run Inference
# =========================

results = model.predict(
    source=str(SOURCE_DIR),
    conf=0.25,
    imgsz=640,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(OUTPUT_DIR),
    name="ppe_prediction",
    exist_ok=True
)

print("\nInference completed successfully!")

print(f"\nResults saved in:\n{OUTPUT_DIR / 'ppe_prediction'}")