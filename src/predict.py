"""
Run batch inference over a directory of images.

Usage:
    python -m src.predict
    python -m src.predict --source path/to/images --conf 0.4
"""

import argparse
from pathlib import Path
from typing import Any

from src.utils import load_yaml, project_path

DEFAULT_SOURCE = "tests/test_img"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch PPE detection over a folder of images."
    )
    parser.add_argument("--source", default=None, help="Image file or directory.")
    parser.add_argument("--model", default=None, help="Path to weights.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    return parser.parse_args()


def run(
    source: str | Path = DEFAULT_SOURCE,
    model_path: str | Path = "models/best.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
) -> Path:
    """Run inference on `source` and return the output directory."""
    from ultralytics import YOLO

    from src.utils import resolve_device

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = project_path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source not found: {source_path}\n"
            "Add some images under tests/test_img, or pass --source."
        )

    weights = Path(model_path)
    if not weights.is_absolute():
        weights = project_path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    output_dir = project_path("runs", "predict")
    YOLO(str(weights)).predict(
        source=str(source_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="ppe_prediction",
        exist_ok=True,
        device=resolve_device(device),
    )
    return output_dir / "ppe_prediction"


def main() -> None:
    args = parse_args()
    defaults: dict[str, Any] = load_yaml(
        project_path("configs", "inference_config.yaml")
    )
    out = run(
        source=args.source or DEFAULT_SOURCE,
        model_path=args.model or defaults.get("model", "models/best.pt"),
        imgsz=args.imgsz or int(defaults.get("imgsz", 640)),
        conf=args.conf if args.conf is not None else float(defaults.get("conf", 0.25)),
        iou=args.iou if args.iou is not None else float(defaults.get("iou", 0.7)),
        device=defaults.get("device"),
    )
    print(f"\nInference complete.\nResults saved in: {out}")


if __name__ == "__main__":
    main()
