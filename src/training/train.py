"""
Fine-tune a YOLO11 checkpoint on the PPE detection dataset.

Configuration lives in configs/train_config.yaml; the dataset definition is in
configs/data.yaml.

Usage:
    python -m src.training.train
"""

from pathlib import Path
from typing import Any

from src.utils import load_yaml, project_path

CONFIG_PATH = "configs/train_config.yaml"


def load_train_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the training configuration."""
    config = load_yaml(project_path(path))
    if not config:
        raise FileNotFoundError(
            f"No training config found at {project_path(path)}. "
            "Expected model, data, epochs, imgsz and batch keys."
        )
    return config


def check_gpu() -> str:
    """
    Report the available accelerator and return the device string to train on.

    This is the only place torch is imported for the training path, so the
    config loader stays importable without it.
    """
    import torch

    print("=" * 46)
    print("GPU check")
    print("=" * 46)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo GPU found. Training on CPU.")
        print("=" * 46)
        return "cpu"

    print(f"GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}")
        print(f"  Name     : {torch.cuda.get_device_name(i)}")
        print(f"  Memory   : {props.total_memory / 1024**3:.2f} GB")
        print(f"  CUDA cap : {props.major}.{props.minor}")
    print("=" * 46)
    return "0"


def train_model(model: Any, config: dict[str, Any], device: str) -> None:
    """Run Ultralytics training with the supplied configuration."""
    model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        workers=config["workers"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        patience=config["patience"],
        project=config["project"],
        name=config["name"],
        cache=config["cache"],
        cos_lr=config["cos_lr"],
        amp=config["amp"],
        plots=config["plots"],
        save=config["save"],
        save_period=config["save_period"],
        seed=config["seed"],
        device=device,
    )


def main() -> None:
    from ultralytics import YOLO

    config = load_train_config()
    # check_gpu both reports the hardware and returns the device string.
    device = check_gpu()

    print(f"Training {config['model']} on {config['data']}")
    model = YOLO(config["model"])
    train_model(model, config, device)

    print(
        "\nTraining finished.\n"
        "Copy the best checkpoint to models/best.pt so the app picks it up:\n"
        f"  runs/{config['project'].split('/')[-1]}/{config['name']}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
