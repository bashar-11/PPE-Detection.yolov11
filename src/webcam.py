import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ======================================
# Project Paths
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# ======================================
# Load Model
# ======================================

print("Loading model...")
model = YOLO(str(MODEL_PATH))

# ======================================
# Open Webcam
# ======================================

cap = cv2.VideoCapture(0)   # لو عندك أكثر من كاميرا جرّب 1 أو 2

if not cap.isOpened():
    raise Exception("Could not open webcam.")

prev_time = time.time()

print("Press 'q' to exit")

# ======================================
# Live Detection
# ======================================

while True:

    success, frame = cap.read()

    if not success:
        break

    # Inference
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.30,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Smart Safety Monitoring", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ======================================
# Release Resources
# ======================================

cap.release()
cv2.destroyAllWindows()