# Offroad Semantic Segmentation Submission Kit

This repository root contains the usable project for the Duality AI offroad segmentation task. The `offroad_ai_system` folder is intentionally ignored here.

## What Is Included

- `Offroad_Segmentation_Scripts/train.py`: config-driven training entrypoint
- `Offroad_Segmentation_Scripts/test.py`: evaluation and prediction export entrypoint
- `Offroad_Segmentation_Scripts/visualize.py`: raw-mask or class-mask colorizer
- `Offroad_Segmentation_Scripts/configs/default_config.json`: default runtime config
- `Offroad_Segmentation_Scripts/runs/`: generated checkpoints, plots, metrics, and evaluation outputs
- `frontend/`: styled React dashboard for browsing metrics and live sample comparisons
- `REPORT_TEMPLATE.md`: report structure aligned with the hackathon PDF

## Dataset Layout Expected By The Scripts

The scripts assume the dataset folders exactly as they exist in this workspace:

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

The mask contract is fixed to these 10 raw Falcon values:

- `100` Trees
- `200` Lush Bushes
- `300` Dry Grass
- `500` Dry Bushes
- `550` Ground Clutter
- `600` Flowers
- `700` Logs
- `800` Rocks
- `7100` Landscape
- `10000` Sky

## Environment Setup

### Option 1: existing Python environment

Install the dependencies used by this code path:

```powershell
python -m pip install -r .\Offroad_Segmentation_Scripts\requirements.txt
```

### Option 2: Conda helper scripts

```powershell
cd .\Offroad_Segmentation_Scripts\ENV_SETUP
.\setup_env.bat
```

The helper scripts install CPU-safe defaults. If you want GPU support, install the matching CUDA PyTorch build after environment creation.

## Exact Commands

### 1. Training smoke test

```powershell
python .\Offroad_Segmentation_Scripts\train.py --dry_run --max_train_batches 1 --max_val_batches 1 --run_name smoke_test
```

### 2. Full training run

```powershell
python .\Offroad_Segmentation_Scripts\train.py --run_name dinov2_baseline
```

### 3. Evaluate on validation data

```powershell
python .\Offroad_Segmentation_Scripts\test.py --model_path .\Offroad_Segmentation_Scripts\runs\dinov2_baseline\checkpoints\best_iou.pth --data_root .\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val
```

### 4. Evaluate on test data

```powershell
python .\Offroad_Segmentation_Scripts\test.py --model_path .\Offroad_Segmentation_Scripts\runs\dinov2_baseline\checkpoints\best_iou.pth --data_root .\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages
```

### 5. Colorize raw Falcon masks or predicted class masks

```powershell
python .\Offroad_Segmentation_Scripts\visualize.py --input_path .\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation --mode raw
```

### 6. Export assets for the React dashboard

```powershell
python .\frontend\scripts\export_dashboard_assets.py --run-dir .\Offroad_Segmentation_Scripts\runs\quick_cpu_100x20
```

### 7. Start the API server for live run switching and image upload inference

```powershell
.\.venv\Scripts\uvicorn frontend.server.app:app --reload --port 8000
```

### 8. Start the React dashboard

```powershell
cd .\frontend
npm install
npm run dev
```

## Output Directory Map

Training outputs are written under:

```text
Offroad_Segmentation_Scripts/runs/<run_name>/
  checkpoints/
    best_iou.pth
    last.pth
  metrics/
    history.json
    history.csv
    training_summary.json
  plots/
    training_metrics.png
  config_resolved.json
  data_contract.json
```

Evaluation outputs are written under the run-linked evaluation folder unless `--output_dir` is provided:

```text
Offroad_Segmentation_Scripts/runs/<run_name>/evaluations/<split_name>/
  predictions/
    raw_masks/
    color_masks/
  comparisons/
  worst_cases/
  evaluation_metrics.txt
  evaluation_metrics.json
  per_class_iou.png
  per_class_dice.png
  confusion_matrix.png
  per_image_metrics.csv
  config_resolved.json
  data_contract.json
```

## Reproducing Final Results

1. Install dependencies.
2. Run `train.py` with a named run.
3. Use `best_iou.pth` from that run for validation and test evaluation.
4. Collect plots and metrics from the run directory.
5. Fill `REPORT_TEMPLATE.md` with the final screenshots, scores, and failure-case analysis.

## Notes

- The scripts never use the test folder for training.
- Dataset raw values are validated before training and evaluation.
- The default config is CPU-safe: frozen DINOv2 backbone, batch size `1`, workers `0`.
- If a run directory already exists, a timestamped suffix is added automatically.
- The React frontend reads exported files from `frontend/public/dashboard`.
- The React frontend can also work in live mode through the FastAPI server at `http://127.0.0.1:8000`.
"# TerrainScope" 
