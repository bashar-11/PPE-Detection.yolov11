from pathlib import Path
from ultralytics import YOLO
import time

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "tests" / "video"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "video_prediction"

# ==========================
# Check Files
# ==========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found:\n{VIDEO_PATH}")

# ==========================
# Load Model
# ==========================

print("Loading model...")
model = YOLO(str(MODEL_PATH))

# ==========================
# Video Inference
# ==========================

print("Running inference...")

start = time.time()

model.predict(
    source=str(VIDEO_PATH),
    imgsz=640,
    conf=0.25,
    save=True,
    save_txt=False,
    save_conf=False,
    show=False,
    project=str(OUTPUT_DIR),
    name="ppe_video",
    exist_ok=True
)

end = time.time()

print("\n=================================")
print("Inference Completed Successfully")
print("=================================")

print(f"Time: {end-start:.2f} sec")

print("\nOutput Folder:")
print(OUTPUT_DIR / "ppe_video")