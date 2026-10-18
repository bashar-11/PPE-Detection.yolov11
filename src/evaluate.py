"""
Evaluate the trained PPE detection model.

Reports precision, recall, mAP@50 and mAP@50-95 on the test split, and writes
the numbers to JSON so they can be quoted in the README instead of being
re-derived by hand each time.

Usage:
    python -m src.evaluate
"""

import json
from pathlib import Path
from typing import Any

from src.utils import (
    format_metrics,
    load_yaml,
    project_path,
    resolve_device,
    summarize_metrics,
)

DEFAULT_MODEL = "models/best.pt"
DEFAULT_DATA = "configs/data.yaml"
OUTPUT_PROJECT = "runs/evaluation"


def evaluate(
    model_path: str | Path = DEFAULT_MODEL,
    data_path: str | Path = DEFAULT_DATA,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
    project: str = OUTPUT_PROJECT,
    name: str = "ppe_eval",
) -> dict[str, float]:
    """Run validation and return the headline metrics as a plain dict."""
    from ultralytics import YOLO

    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = project_path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Model weights not found: {model_file}")

    data_file = Path(data_path)
    if not data_file.is_absolute():
        data_file = project_path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_file}")

    model = YOLO(str(model_file))
    metrics = model.val(
        data=str(data_file),
        imgsz=imgsz,
        batch=batch,
        split="test",
        conf=conf,
        iou=iou,
        save_json=True,
        plots=True,
        project=str(project_path(project)),
        name=name,
        device=resolve_device(device),
    )
    return summarize_metrics(metrics)


def save_metrics(summary: dict[str, float], out_dir: str = OUTPUT_PROJECT) -> Path:
    """Write the metrics summary to <out_dir>/metrics.json and return the path."""
    target = project_path(out_dir) / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metrics": summary, "split": "test"}
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return target


def main() -> None:
    inference = load_yaml(project_path("configs", "inference_config.yaml"))
    summary = evaluate(
        imgsz=int(inference.get("imgsz", 640)),
        conf=float(inference.get("conf", 0.25)),
        iou=float(inference.get("iou", 0.7)),
        device=inference.get("device"),
    )
    print(format_metrics(summary))
    print(f"\nSaved to: {save_metrics(summary)}")


if __name__ == "__main__":
    main()
