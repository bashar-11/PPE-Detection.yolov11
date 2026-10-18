"""
Live PPE detection from the webcam.

Press 'q' in the preview window to exit.

Usage:
    python -m src.webcam
    python -m src.webcam --conf 0.4 --camera 1
"""

import argparse
import time
from pathlib import Path
from typing import Any

from src.utils import load_yaml, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live PPE detection from a webcam.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. 0 is usually the built-in webcam.",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def run(
    camera_index: int = 0,
    model_path: str | Path = "models/best.pt",
    imgsz: int = 640,
    conf: float = 0.30,
    device: str | None = None,
) -> None:
    """
    Open the camera and run detection frame by frame.

    The confidence default here (0.30) is higher than the still-image default
    because video frames carry sensor noise and compression artefacts, which
    otherwise produce spurious detections that flicker in and out.
    """
    import cv2
    from ultralytics import YOLO

    from src.utils import resolve_device

    weights = project_path(model_path)
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    print("Loading model...")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    device_arg = resolve_device(device)
    prev_time = time.time()
    print("Press 'q' in the preview window to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device=device_arg,
            )
            annotated = results[0].plot()

            now = time.time()
            fps = 1 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Smart Safety Monitoring", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release the camera, even if detection raises mid-loop,
        # otherwise the device stays locked until the process is killed.
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    defaults: dict[str, Any] = load_yaml(
        project_path("configs", "inference_config.yaml")
    )
    run(
        camera_index=args.camera,
        model_path=args.model or defaults.get("model", "models/best.pt"),
        imgsz=int(defaults.get("imgsz", 640)),
        conf=args.conf if args.conf is not None else 0.30,
        device=defaults.get("device"),
    )


if __name__ == "__main__":
    main()
