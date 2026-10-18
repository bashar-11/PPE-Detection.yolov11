"""
Shared helpers for the training, evaluation and inference scripts.

Kept free of heavy imports (torch, ultralytics) so that the test suite can
exercise this logic without loading a model.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config, returning an empty dict for a missing or empty file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def project_path(*parts: str) -> Path:
    """Build a path relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_device(preferred: str | None = None) -> str:
    """
    Pick the device to run inference on.

    An explicit request is honoured, otherwise CUDA is used when available and
    the pipeline falls back to CPU. Importing torch is deferred so that simply
    importing this module stays cheap.
    """
    if preferred:
        return preferred
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def summarize_metrics(metrics: Any) -> Dict[str, float]:
    """
    Pull the four headline detection numbers out of an Ultralytics
    DetMetrics object into a plain dict, so they can be printed, logged and
    written to JSON without holding onto the whole metrics object.
    """
    box = getattr(metrics, "box", None)
    if box is None:
        return {}
    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }


def format_metrics(summary: Dict[str, float]) -> str:
    """Render a metrics summary as an aligned, human-readable block."""
    if not summary:
        return "No metrics returned."
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP@50",
        "map50_95": "mAP@50-95",
    }
    width = max(len(v) for v in labels.values())
    lines = ["=" * 46, "Evaluation results", "=" * 46]
    for key, label in labels.items():
        if key in summary:
            lines.append(f"{label.ljust(width)} : {summary[key]:.4f}")
    lines.append("=" * 46)
    return "\n".join(lines)