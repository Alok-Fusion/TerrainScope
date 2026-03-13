# TerrainScope: Off-Road Semantic Segmentation with DINOv2

> Team Invincible  
> Karna Bhosle, Alok Kushwaha, Shankar Gouda

TerrainScope is a complete off-road semantic segmentation pipeline built for the Duality AI off-road scene understanding task. The repository includes dataset handling, training, evaluation, visualization, a live inference API, and a React dashboard for qualitative comparison and model inspection.

The production code in this repository lives outside the legacy `offroad_ai_system` folder. That older folder is intentionally excluded from the active workflow.

## Problem Statement

Autonomous off-road navigation requires a model that can separate traversable terrain from visually similar obstacles such as bushes, logs, rocks, clutter, and background landscape. The goal of this project is to train a semantic segmentation system that can classify every pixel in an off-road image into one of ten terrain classes and expose the result through both reproducible scripts and an interactive web interface.

## Solution Overview

TerrainScope uses a frozen **DINOv2 ViT-S/14** backbone as a feature extractor and trains a lightweight segmentation head on top of those features. The training and evaluation pipeline is config-driven, CPU-safe by default, and backed by a React + FastAPI demo layer for live inference and visual comparison.

### Key Capabilities

- 10-class off-road semantic segmentation with a fixed `uint16` Falcon mask contract
- Config-driven `train.py`, `test.py`, and `visualize.py` entrypoints
- Strict dataset validation to prevent silent label corruption
- Training and evaluation artifact export: checkpoints, metrics, plots, confusion matrix, prediction masks, comparisons, and worst cases
- FastAPI backend for run switching and image upload inference
- React dashboard for metrics, side-by-side comparison, fullscreen viewing, and image-level suggestions

## Final Results Snapshot

The best published run in this repository is `quick_cpu_100x20`. That run was resumed once and trained for **40 total epochs** while preserving the same run folder and checkpoint lineage.

| Split | Mean IoU | Mean Dice | Pixel Accuracy | Images |
| --- | ---: | ---: | ---: | ---: |
| Validation | **30.30%** | **38.88%** | **75.35%** | 317 |
| Test | **21.08%** | **26.27%** | **63.15%** | 1002 |

### Training Summary From Final Run

| Metric | Value |
| --- | ---: |
| Best validation IoU during training | 26.24% |
| Final training IoU | 30.63% |
| Final validation IoU during training loop | 26.24% |
| Final training Dice | 39.13% |
| Final validation Dice during training loop | 31.59% |
| Final training pixel accuracy | 77.50% |
| Final validation pixel accuracy during training loop | 68.61% |

## Dataset

### Download Link

- Google Drive dataset: <https://drive.google.com/drive/u/0/folders/1TEyTM514of7vPAC0evBV6IdgX9_v5PV3>

### Expected Folder Layout

```text
Offroad_Segmentation_Training_Dataset/
  Offroad_Segmentation_Training_Dataset/
    train/
      Color_Images/
      Segmentation/
    val/
      Color_Images/
      Segmentation/

Offroad_Segmentation_testImages/
  Offroad_Segmentation_testImages/
    Color_Images/
    Segmentation/
```

### Dataset Split Sizes

| Split | Images |
| --- | ---: |
| Train | 2857 |
| Validation | 317 |
| Test | 1002 |

### Label Contract

All masks are read as `uint16` and mapped through a shared canonical class contract.

| Raw Value | Class ID | Class Name |
| ---: | ---: | --- |
| 100 | 0 | Trees |
| 200 | 1 | Lush Bushes |
| 300 | 2 | Dry Grass |
| 500 | 3 | Dry Bushes |
| 550 | 4 | Ground Clutter |
| 600 | 5 | Flowers |
| 700 | 6 | Logs |
| 800 | 7 | Rocks |
| 7100 | 8 | Landscape |
| 10000 | 9 | Sky |

Unknown mask values are mapped to `255` and treated as invalid. This prevents silent background remapping and forces bad annotations to fail early.

## Architecture

```mermaid
flowchart LR
    A[RGB Image] --> B[Resize + Normalize]
    B --> C[Frozen DINOv2 ViT-S/14 Backbone]
    C --> D[Feature Upsampling + Segmentation Head]
    D --> E[10-Class Logits]
    E --> F[Argmax Segmentation Mask]
    F --> G[Metrics and Visualization]
    F --> H[FastAPI Inference Service]
    H --> I[React Dashboard]
```

### Model Components

| Component | Choice |
| --- | --- |
| Supported model types | `dinov2`, `segformer_b0`, `deeplabv3plus` |
| Default backbone | `dinov2_vits14` |
| Alternate DINOv2 backbone | `dinov2_vitb14` |
| SegFormer variant | `nvidia/segformer-b0-finetuned-ade-512-512` |
| DeepLabV3+ encoder | `mobilenet_v2` |
| Encoder mode | Frozen during training by default |
| Head | Lightweight segmentation head |
| Input size for final run | `140 x 252` |
| Default device | Auto-detect, CPU-safe by default |
| Loss | Cross-entropy on class IDs |
| Optimizer | SGD with momentum |

### Why This Design

- **DINOv2** provides strong semantic features even when compute is limited.
- **Frozen backbone training** keeps the project reproducible on CPU-only setups.
- **Shared dataset contract** eliminates label mismatches between training, evaluation, visualization, and live inference.
- **Separate API + frontend** makes the project easier to demo for judges and easier to reproduce for reviewers.

## Repository Structure

```text
.
+-- Offroad_Segmentation_Scripts/
�   +-- configs/
�   +-- ENV_SETUP/
�   +-- offroad_segmentation/
�   +-- train.py
�   +-- test.py
�   +-- visualize.py
�   +-- requirements.txt
+-- frontend/
�   +-- server/
�   +-- src/
�   +-- scripts/
+-- docs/
�   +-- screenshots/
+-- README.md
+-- report.md
+-- report.pdf
+-- project_documentation.txt
```

## Screenshots and Visual Evidence

### Training Curves

The final run shows steady improvement across loss, IoU, Dice, and pixel accuracy over the resumed 40-epoch training history.

![Training curves](docs/screenshots/training_metrics.png)

### Per-Class Validation IoU

This chart highlights which terrain categories the model handles well and which ones remain weak. Landscape and sky are strong; rare obstacle classes remain difficult.

![Per-class IoU](docs/screenshots/per_class_iou.png)

### Validation Confusion Matrix

The confusion matrix shows that visually dominant classes are learned reasonably well, while rare or visually overlapping obstacle categories still confuse the model.

![Validation confusion matrix](docs/screenshots/confusion_matrix.png)

### Qualitative Comparison

Example validation output showing RGB input, ground truth, and model prediction side by side.

![Validation comparison](docs/screenshots/val_comparison.png)

### Failure Case

Representative hard case from the exported worst-case set. This is useful for explaining where the model still struggles.

![Failure case](docs/screenshots/failure_case.png)

## Training and Evaluation Workflow

### 1. Install Dependencies

```powershell
python -m pip install -r .\Offroad_Segmentation_Scripts\requirements.txt
```

### 2. Run a Smoke Test

```powershell
python .\Offroad_Segmentation_Scripts\train.py --dry_run --max_train_batches 1 --max_val_batches 1 --run_name smoke_test
```

### 3. Train a Full Run

```powershell
python .\Offroad_Segmentation_Scripts\train.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu.json --epochs 20 --max_train_batches 100 --max_val_batches 20 --run_name quick_cpu_100x20
```

### Fast model-switching commands

These configs keep training short on CPU by using frozen encoders, batch size `1`, and small image sizes.

```powershell
# DINOv2 ViT-B/14
python .\Offroad_Segmentation_Scripts\train.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu_dinov2_vitb14.json --max_train_batches 50 --max_val_batches 10 --run_name quick_dino_vitb14

# SegFormer-B0
python .\Offroad_Segmentation_Scripts\train.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu_segformer_b0.json --max_train_batches 50 --max_val_batches 10 --run_name quick_segformer_b0

# DeepLabV3+ with MobileNetV2
python .\Offroad_Segmentation_Scripts\train.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu_deeplabv3plus.json --max_train_batches 50 --max_val_batches 10 --run_name quick_deeplabv3plus
```

### 4. Resume Training

```powershell
python .\Offroad_Segmentation_Scripts\train.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu.json --resume_from .\Offroad_Segmentation_Scripts\runs\quick_cpu_100x20\checkpoints\last.pth --epochs 20 --max_train_batches 100 --max_val_batches 20
```

### 5. Evaluate on Validation

```powershell
python .\Offroad_Segmentation_Scripts\test.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu.json --model_path .\Offroad_Segmentation_Scripts\runs\quick_cpu_100x20\checkpoints\best_iou.pth --data_root .\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val
```

### 6. Evaluate on Test

```powershell
python .\Offroad_Segmentation_Scripts\test.py --config .\Offroad_Segmentation_Scripts\configs\quick_cpu.json --model_path .\Offroad_Segmentation_Scripts\runs\quick_cpu_100x20\checkpoints\best_iou.pth --data_root .\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages
```

### 7. Colorize Raw or Predicted Masks

```powershell
python .\Offroad_Segmentation_Scripts\visualize.py --input_path .\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation --mode raw
```

## Live Inference Dashboard

The repository also includes a demo layer for live image upload and run-aware inference.

### Frontend Dashboard Preview

This screenshot shows the tracked dashboard view used to review the final run, inspect metrics, and browse validation outputs.

![TerrainScope frontend dashboard](docs/screenshots/frontend_dashboard.png)

### Start the API Server

```powershell
.\.venv\Scripts\uvicorn frontend.server.app:app --reload --port 8000
```

### Start the React Frontend

```powershell
cd .\frontend
npm install
npm run dev
```

### Frontend Features

- Run switching without manual re-export
- Validation/test comparison browsing
- Fullscreen compare modal
- Upload image inference with backend prediction
- Image metadata, confidence summary, dominant class, and terrain suggestions

## Output Artifacts

A completed run produces the following important outputs:

```text
Offroad_Segmentation_Scripts/runs/<run_name>/
+-- checkpoints/
�   +-- best_iou.pth
�   +-- last.pth
+-- metrics/
�   +-- history.json
�   +-- history.csv
�   +-- training_summary.json
+-- plots/
�   +-- training_metrics.png
+-- evaluations/<split_name>/
    +-- comparisons/
    +-- worst_cases/
    +-- predictions/raw_masks/
    +-- predictions/color_masks/
    +-- confusion_matrix.png
    +-- per_class_iou.png
    +-- per_class_dice.png
    +-- per_image_metrics.csv
    +-- evaluation_metrics.json
    +-- evaluation_metrics.txt
```

## Engineering Fixes Implemented

This repository is not just a model checkpoint dump. The following engineering issues were corrected to make the project reproducible and submission-ready:

- Replaced inconsistent scripts with shared core modules under `Offroad_Segmentation_Scripts/offroad_segmentation`
- Fixed mask loading to preserve `uint16` labels
- Removed the fake background class and enforced a strict 10-class contract
- Corrected dataset root resolution for nested training and test folders
- Added evaluation artifact export and failure-case generation
- Added resume support for training continuation
- Added a FastAPI inference server and React dashboard
- Added report-ready plots, comparisons, and documentation assets

## Reproducibility Notes

- The active workflow **does not use** the legacy `offroad_ai_system` directory.
- The pipeline is CPU-safe by default.
- The backbone is loaded from cached DINOv2 weights when available.
- Final README screenshots are copied into `docs/screenshots/` so they remain visible even though `runs/` is ignored in git.

## Team

**Team Invincible**

- Karna Bhosle
- Alok Kushwaha
- Shankar Gouda

## Additional Documentation

- [Detailed report](report.md)
- [Detailed PDF report](report.pdf)
- [Technical project notes](project_documentation.txt)

## License / Usage Note

This repository was prepared as a hackathon submission and demonstration project for off-road semantic segmentation research and evaluation.
