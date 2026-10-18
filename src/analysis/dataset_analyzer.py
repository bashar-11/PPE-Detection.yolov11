"""
Summarise the PPE detection dataset: class list, image and label counts per split.

Usage:
    python -m src.analysis.dataset_analyzer
"""

import sys
from pathlib import Path
from typing import Any

import yaml

from src.utils import project_path

DATASET_PATH = "data/raw/ppe_dataset"
YAML_FILE = "configs/data.yaml"

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


class DatasetAnalyzer:
    """Read a YOLO-format dataset and report its shape."""

    def __init__(
        self, dataset_path: str | Path = DATASET_PATH, yaml_file: str | Path = YAML_FILE
    ) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_absolute():
            self.dataset_path = project_path(self.dataset_path)

        yaml_path = Path(yaml_file)
        if not yaml_path.is_absolute():
            yaml_path = project_path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset config not found: {yaml_path}\n"
                "See README.dataset.md for how to obtain the dataset."
            )

        with open(yaml_path, "r", encoding="utf-8") as fh:
            self.data: dict[str, Any] = yaml.safe_load(fh)

        self.splits = ("train", "valid", "test")

    def count_images(self, folder: Path) -> int:
        """Count image files across the common extensions."""
        if not folder.exists():
            return 0
        return sum(len(list(folder.glob(ext))) for ext in IMAGE_EXTS)

    def count_labels(self, folder: Path) -> int:
        """Count YOLO label files."""
        if not folder.exists():
            return 0
        return len(list(folder.glob("*.txt")))

    def summary(self) -> dict[str, dict[str, int]]:
        """Return per-split image and label counts."""
        result: dict[str, dict[str, int]] = {}
        for split in self.splits:
            images = self.count_images(self.dataset_path / split / "images")
            labels = self.count_labels(self.dataset_path / split / "labels")
            result[split] = {"images": images, "labels": labels}
        return result

    def print_summary(self) -> None:
        print("=" * 46)
        print("PPE DETECTION DATASET REPORT")
        print("=" * 46)
        print(f"Root: {self.dataset_path}")
        print(f"Classes: {self.data.get('nc', 'unknown')}")
        print()

        names = self.data.get("names", {})
        for idx, name in sorted(names.items(), key=lambda kv: int(kv[0])):
            print(f"  {idx} -> {name}")
        print()

        print(f"{'Split':<8}{'Images':>10}{'Labels':>10}")
        for split, counts in self.summary().items():
            print(f"{split:<8}{counts['images']:>10}{counts['labels']:>10}")
        print("=" * 46)


def main() -> None:
    try:
        DatasetAnalyzer().print_summary()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
