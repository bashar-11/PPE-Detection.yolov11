from ultralytics import YOLO
import yaml
from pathlib import Path
# PROJECT_ROOT = Path("C:\Users\Admin\Desktop\Deep Learning\project\PPE Detection.yolov8\configs\train_config.yaml").resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"



def load_config():

    with open(CONFIG_PATH, "r") as f:

        config = yaml.safe_load(f)

    return config



def create_model(model_name):

    model = YOLO(model_name)

    return model





# ####################################################
# الهدف من الكود ده بس يختار device المناسب
######################################################

import torch


def check_gpu():

    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(f"GPU Count       : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):

            print(f"\nGPU {i}")

            print(f"Name            : {torch.cuda.get_device_name(i)}")

            props = torch.cuda.get_device_properties(i)

            print(f"Memory          : {props.total_memory / 1024**3:.2f} GB")

            print(f"CUDA Capability : {props.major}.{props.minor}")

        device = 0

    else:

        print("\nNo GPU Found. Using CPU.")

        device = "cpu"

    print("=" * 60)

    return device

##################################################################################################

def train_model(model, config, device):

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
        device=device

    )




def main():

    device = check_gpu()

    config = load_config()

    model = create_model(config["model"])

    train_model(model, config, device)



if __name__ == "__main__":

    main()


