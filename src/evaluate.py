from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/best.pt"
DATA_PATH = "data/raw/ppe_dataset/data.yaml"

model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA_PATH,
    imgsz=640,
    batch=16,
    split="test",          # أو val
    conf=0.25,
    iou=0.7,
    save_json=True,
    plots=True,
    project="runs/evaluation",
    name="ppe_eval"
)

print("="*60)
print("Evaluation Results")
print("="*60)

print(f"Precision      : {metrics.box.mp:.4f}")
print(f"Recall         : {metrics.box.mr:.4f}")
print(f"mAP@50         : {metrics.box.map50:.4f}")
print(f"mAP@50-95      : {metrics.box.map:.4f}")

print("="*60)
