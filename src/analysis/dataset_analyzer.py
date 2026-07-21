from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ppe_dataset"

YAML_FILE = DATASET_PATH / "data.yaml"


class DatasetAnalyzer:

    def __init__(self):

        with open(YAML_FILE, "r") as f:
            self.data = yaml.safe_load(f)

    def count_images(self, folder):

        exts = ("*.jpg", "*.jpeg", "*.png")

        total = 0

        for ext in exts:

            total += len(list(folder.glob(ext)))

        return total

    def count_labels(self, folder):

        return len(list(folder.glob("*.txt")))

    def print_summary(self):

        print("=" * 60)

        print("SMART SAFETY DATASET REPORT")

        print("=" * 60)

        print()

        print(f"Classes : {self.data['nc']}")

        print()

        print("Class Names")

        for idx, name in self.data["names"].items():

            print(f"{idx} -> {name}")

        print()

        train = self.count_images(DATASET_PATH / "train/images")
        valid = self.count_images(DATASET_PATH / "valid/images")
        test = self.count_images(DATASET_PATH / "test/images")

        print("Images")

        print(f"Train : {train}")

        print(f"Valid : {valid}")

        print(f"Test  : {test}")

        print()

        train_l = self.count_labels(DATASET_PATH / "train/labels")
        valid_l = self.count_labels(DATASET_PATH / "valid/labels")
        test_l = self.count_labels(DATASET_PATH / "test/labels")

        print("Labels")

        print(f"Train : {train_l}")

        print(f"Valid : {valid_l}")

        print(f"Test  : {test_l}")

        print()

        print("=" * 60)


if __name__ == "__main__":

    analyzer = DatasetAnalyzer()

    analyzer.print_summary()