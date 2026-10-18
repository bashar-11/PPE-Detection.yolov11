# Dataset: PPE Detection

This project trains on the **PPE Detection** dataset from Roboflow Universe.
**The dataset is not committed to this repository.**

## Source

| Field | Value |
|---|---|
| Dataset page | <https://universe.roboflow.com/b-abdelraziq/ppe-detection-vwjor-aqptb> |
| Provided by | A Roboflow Universe user (`b-abdelraziq`) |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Format | YOLO-style bounding box annotations (YOLOv8 / YOLOv11 compatible) |
| Size | ~10,000 annotated images |
| Preprocessing | None applied at export (raw export) |
| Classes | `Reflective-Jacket`, `Safety-Helmet` |

## Why it is not in the repository

- It is roughly **20,000 files**. Committing it made a clone take minutes for
  every visitor.
- It carries its own licence and attribution requirements, which are cleaner to
  honour by referencing the source than by mirroring it.
- It is downloadable, so storing it in git adds size without adding access.

A CI job fails the build if the dataset reappears under `data/`.

## Getting it

1. Open the dataset page linked above and download the **YOLO** export.
2. Extract it so that the dataset root ends up at:

   ```
   data/raw/ppe_dataset/
   ├── train/{images,labels}
   ├── valid/{images,labels}
   ├── test/{images,labels}
   └── data.yaml
   ```

3. Check it landed correctly:

   ```bash
   python -m src.analysis.dataset_analyzer
   ```

The class definition also lives in [`configs/data.yaml`](configs/data.yaml) in
this repository, so if the export's own `data.yaml` is missing you can point
`configs/data.yaml` at your extracted directory instead.

## Attribution

The CC BY 4.0 licence requires attribution. If you reuse this dataset or
anything trained on it, credit the original contributor and Roboflow, and
include a link back to the dataset page.

## Related files

- [`configs/data.yaml`](configs/data.yaml) — class names and split paths used by
  training and evaluation.
- [`configs/train_config.yaml`](configs/train_config.yaml) — training
  hyperparameters.
- [`src/analysis/dataset_analyzer.py`](src/analysis/dataset_analyzer.py) — reports
  class list and per-split image/label counts.

---

*This file documents the dataset used by
[PPE-Detection.yolov11](README.md). See the [main README](README.md) for the
full project overview.*