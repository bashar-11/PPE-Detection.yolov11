"""
Run detection over a video file and write an annotated copy.

Usage:
    python -m src.predict_video
    python -m src.predict_video --source path/to/clip.mp4
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from src.utils import load_yaml, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPE detection over a video file.")
    parser.add_argument("--source", default=None, help="Video file or directory.")
    parser.add_argument("--model", default=None, help="Path to weights.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    return parser.parse_args()


def run(
    source: str | Path,
    model_path: str | Path = "models/best.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = None,
) -> Dict[str, Any]:
    """Process a video and return the output directory plus elapsed time."""
    from ultralytics import YOLO

    from src.utils import resolve_device

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = project_path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    weights = Path(model_path)
    if not weights.is_absolute():
        weights = project_path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    print("Loading model...")
    model = YOLO(str(weights))

    print("Running inference...")
    start = time.time()
    model.predict(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        save=True,
        save_txt=False,
        save_conf=False,
        show=False,
        project=str(project_path("runs", "video_prediction")),
        name="ppe_video",
        exist_ok=True,
        device=resolve_device(device),
    )
    elapsed = time.time() - start

    out = project_path("runs", "video_prediction", "ppe_video")
    print("=" * 46)
    print(f"Inference complete in {elapsed:.2f}s")
    print(f"Output: {out}")
    print("=" * 46)
    return {"output_dir": str(out), "elapsed_seconds": elapsed}


def main() -> None:
    args = parse_args()
    defaults: Dict[str, Any] = load_yaml(project_path("configs", "inference_config.yaml"))
    source = args.source or defaults.get("source")
    if not source:
        raise SystemExit(
            "No video specified. Pass --source path/to/video.mp4\n"
            "(test videos are not committed to this repository)."
        )
    run(
        source=source,
        model_path=args.model or defaults.get("model", "models/best.pt"),
        imgsz=args.imgsz or int(defaults.get("imgsz", 640)),
        conf=args.conf if args.conf is not None else float(defaults.get("conf", 0.25)),
        device=defaults.get("device"),
    )


if __name__ == "__main__":
    main()