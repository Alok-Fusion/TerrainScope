# TerrainScope
## Off-Road Semantic Segmentation and Live Inference Dashboard

**Team Name:** Team Invincible  
**Team Members:** Karna Bhosle, Alok Kushwaha, Shankar Gouda  
**Project Focus:** Off-road semantic segmentation for terrain understanding  
**Tagline:** A reproducible DINOv2 baseline with evaluation exports, failure analysis, and live review

---

## Title & Summary

TerrainScope is an end-to-end off-road segmentation system. It predicts one of 10 classes per pixel, evaluates results with IoU, Dice, and pixel accuracy, exports review artifacts, and provides a frontend for live inference and scene analysis.

This submission was designed to be:
- clear,
- reproducible,
- CPU-practical during development,
- and straightforward for judges to review.

### Final Headline Results

Two DINOv2 runs define the current project state:

| Run | Role | Validation IoU | Validation Dice | Validation Accuracy | Test IoU | Test Dice | Test Accuracy |
|-------|-------------|---------:|----------:|--------------:|---------:|----------:|--------------:|
| `quick_cpu_100x20` | Best balanced CPU baseline | 32.93% | 41.75% | 77.00% | **21.21%** | **26.34%** | **61.66%** |
| `probe_continue_high_iou` | Best validation-focused continuation | **42.88%** | **55.42%** | **78.38%** | 19.80% | 25.84% | 40.71% |

**Best Validation Model:** `probe_continue_high_iou` with 42.88% validation IoU  
**Best Test-Generalizing Model:** `quick_cpu_100x20`

---

## 1. Problem

Off-road scenes are difficult to segment because:
- terrain classes overlap visually,
- vegetation and clutter blur boundaries,
- small obstacle classes are easy to miss,
- and Falcon masks use raw values instead of simple class IDs.

The system needed to solve four things:
1. predict 10 semantic classes per pixel,
2. keep the pipeline reproducible,
3. export strong evaluation evidence,
4. support live visual inspection.

### Dataset Summary

| Split | Images | Path |
| --- | ---: | --- |
| Train | 2857 | `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train` |
| Validation | 317 | `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val` |
| Test | 1002 | `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages` |

### 10-Class Label Contract

| Raw Value | Class |
| ---: | --- |
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape |
| 10000 | Sky |

Critical engineering fix:
- masks are read as raw `uint16`,
- mapped through one strict shared label contract,
- and unknown values are sent to `ignore_index = 255`.

### Verified Raw Values by Split

| Split Group | Raw Values Found |
| --- | --- |
| Train + Validation | `[100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]` |
| Test | `[100, 200, 300, 500, 800, 7100, 10000]` |

Interpretation:
- train and validation contain the full supported label contract,
- test contains a valid subset of supported classes,
- and no unsupported raw values were found in the verified scan.

---

## 2. Methodology

### 2.1 Training Pipeline

1. Load RGB images and Falcon masks from the actual nested dataset structure.
2. Read masks unchanged to preserve the original raw values.
3. Convert raw mask values into class IDs `0..9`.
4. Resize RGB images and normalize them according to model requirements.
5. Resize masks with nearest-neighbor interpolation.
6. Process through selected segmentation model (DINOv2, SegFormer, or DeepLabV3+).
7. Train with either baseline cross-entropy or the stronger continuation recipe using class-balanced cross-entropy + focal + dice loss.
8. Save checkpoints, plots, comparisons, confusion matrices, and worst cases.

### 2.2 Architecture Overview

The system supports three segmentation backends for comparison, but the strongest validated results in this report come from DINOv2 with a warm-start continuation stage:

```text
Input RGB Image + Falcon Mask
        |
        v
Preprocessing
- resize
- normalize
- raw-to-class conversion
        |
        +---------------------+
        |                     |
        v                     v
   DINOv2 ViT-B/14       SegFormer B0
   (Frozen backbone)     (Pretrained)
        |                     |
        +---------------------+
        |
        v
   DeepLabV3+ MobileNetV2
   (Pretrained backbone)
        |
        v
   10-Class Segmentation Output
        |
        +--> Metrics
        +--> Checkpoints
        +--> Evaluation exports
        +--> Live API + React dashboard
```

### 2.3 Model Configurations

| Run / Model | Architecture | Backbone | Input Size | Pretrained | Frozen |
|-------|-------------|----------|------------|------------|--------|
| `quick_cpu_100x20` | Transformer | DINOv2 ViT-S/14 | 140x252 | Yes | Yes |
| `probe_continue_high_iou` | Transformer | DINOv2 ViT-S/14 | 224x392 | Yes | Yes |
| SegFormer B0 comparison | Transformer | nvidia/segformer-b0-finetuned-ade-512-512 | 140x252 | Yes | No |
| DeepLabV3+ comparison | CNN | deeplabv3plus/mobilenet_v2 | 144x256 | Yes | No |

### 2.4 Training Setup

| Item | Baseline | High-IoU Continuation |
| --- | --- | --- |
| Batch size | `1` | `1` |
| Accumulation | `1` | `2` |
| Optimizer | SGD | AdamW |
| Scheduler | none | cosine |
| Loss | Cross-entropy | Class-balanced CE + Focal + Dice |
| Sampling | standard shuffle | balanced sampling |
| Augmentation | off | flip + crop + color jitter |

Why this staged approach:
- the baseline remains the fastest CPU-safe path
- the continuation reuses the best baseline checkpoint instead of restarting
- stronger loss and sampling target the rare classes that hold the mean IoU down
- alternate models remain useful for comparison, but DINOv2 stayed strongest in the validated runs

---

## 3. Results & Performance Metrics

### 3.1 Overall Results

The most important comparison is now between the original DINOv2 baseline and the stronger warm-start continuation:

| Run | Mean IoU | Mean Dice | Pixel Accuracy | Notes |
|-------|----------:|-----------:|----------------:|--------------|
| `quick_cpu_100x20` validation | 32.93% | 41.75% | 77.00% | Best baseline before continuation |
| `probe_continue_high_iou` validation | **42.88%** | **55.42%** | **78.38%** | Best validation score in the repo |
| `quick_cpu_100x20` test | **21.21%** | **26.34%** | **61.66%** | Best test-generalizing run |
| `probe_continue_high_iou` test | 19.80% | 25.84% | 40.71% | Validation improvement did not transfer yet |

**Key Findings:**
- the warm-start continuation improved validation IoU by **9.95 percentage points**
- the same continuation hurt held-out test performance, so it is not yet the best deployment candidate
- DINOv2 remained stronger than the short-run SegFormer B0 and DeepLabV3+ comparisons in this codebase

### 3.2 Training-Time Validation vs Full Validation

The training loop used capped validation batches for faster feedback. The standalone evaluation pass used the full validation set and is the correct number to report publicly.

| Validation View | Mean IoU | Mean Dice | Pixel Accuracy | Notes |
| --- | ---: | ---: | ---: | --- |
| Baseline training-time summary | 27.06% | 32.24% | 70.15% | Faster, capped batches |
| Baseline full validation evaluation | 32.93% | 41.75% | 77.00% | Original CPU baseline |
| Continuation training-time summary | 34.76% | 43.48% | 73.29% | Best capped continuation checkpoint |
| Continuation full validation evaluation | **42.88%** | **55.42%** | **78.38%** | Best current validation metric |

### 3.2 Per-Class Performance Analysis

Detailed per-class performance metrics are available in the evaluation outputs for each trained model. The models show typical segmentation behavior:

- **Best Performance:** Large, visually distinct classes (Sky, Landscape, Dry Grass)
- **Challenging Classes:** Small obstacles and rare categories (Logs, Rocks, Ground Clutter, Dry Bushes)
- **Architecture Comparison:** Transformer models (SegFormer, DINOv2) show better overall performance than CNN-based DeepLabV3+

For detailed per-class IoU, Dice scores, and confusion matrices, see:
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/`
- `Offroad_Segmentation_Scripts/runs/probe_continue_high_iou/evaluations/val_full_epoch8/`

### 3.3 Required Charts and Visualizations

Each trained model generates comprehensive evaluation artifacts including:

#### Per-Class IoU Charts
Available for each model:
- SegFormer B0: `Offroad_Segmentation_Scripts/runs/segformer_b0_40ep/evaluations/val/per_class_iou.png`
- DINOv2 ViT-B/14: `Offroad_Segmentation_Scripts/runs/dino_vitb14_40ep_20260313_050603/evaluations/val/per_class_iou.png`
- DeepLabV3+: `Offroad_Segmentation_Scripts/runs/deeplabv3plus_40ep_20260313_051758/evaluations/val/per_class_iou.png`

#### Confusion Matrices
Generated for all models showing prediction distributions across classes.

#### Prediction Visualizations
Color-coded segmentation masks, side-by-side comparisons, and failure case analysis available in each model's evaluation folder.

### 3.4 Test Set Evaluation

Test set evaluation was performed on the original DINOv2 model. The current multi-model implementation provides validation results for all three architectures. Test evaluation can be run using:

```bash
python Offroad_Segmentation_Scripts/test.py --config Offroad_Segmentation_Scripts/configs/quick_cpu_[model].json --model_path [checkpoint_path] --data_root Offroad_Segmentation_testImages/Offroad_Segmentation_testImages
```

Where `[model]` is one of: `dinov2_vitb14`, `segformer_b0`, or `deeplabv3plus`

### 3.5 Representative Prediction Examples

Prediction visualizations, comparison images, and failure case analysis are available in each model's evaluation folder:

- **SegFormer B0:** `Offroad_Segmentation_Scripts/runs/segformer_b0_40ep/evaluations/val/comparisons/`
- **DINOv2 ViT-B/14:** `Offroad_Segmentation_Scripts/runs/dino_vitb14_40ep_20260313_050603/evaluations/val/comparisons/`
- **DeepLabV3+:** `Offroad_Segmentation_Scripts/runs/deeplabv3plus_40ep_20260313_051758/evaluations/val/comparisons/`

These folders contain:
- Color-coded segmentation predictions
- Side-by-side input/mask/prediction comparisons
- Failure case analysis for worst-performing samples

The visualizations show that all models capture broad scene structure well but struggle with small obstacle classes, which is typical for segmentation models on complex off-road scenes.

---

## 4. Challenges & Solutions

| Challenge | What Went Wrong | Solution Applied | Result |
| --- | --- | --- | --- |
| Raw Falcon masks | Labels could be corrupted by naive reading | Read masks unchanged and enforce a strict 10-class mapping | Stable labels and training |
| Dataset path mismatch | Scripts did not match the real folder structure | Standardized the true nested train/val/test paths | Reproducible loading |
| CPU training cost | Heavy models were too slow locally | Frozen DINOv2 + lightweight head + quick CPU config | Feasible local training |
| Long runs were hard to finish | One short run was not enough | Added checkpoint resume support | Reached 40 recorded epochs |
| Weak presentation quality | Static metrics alone were not enough | Added evaluation exports, API backend, and React frontend | Better demo and review flow |

### Example Entry

- **Task:** High-IoU continuation from the best baseline checkpoint  
- **Initial IoU Score:** 32.93% validation IoU  
- **Issue Faced:** Rare classes still suppressed the mean IoU, but full retraining on CPU was too slow.  
- **Solution:** Warm-start from the baseline checkpoint and switch to class-balanced CE + Focal + Dice loss, balanced sampling, and stronger augmentation.  
- **New Score:** Validation IoU improved to **42.88%**.  

### 4.1 What Was Improved in the Engineering Stack

Beyond model training, the project was upgraded into a full submission kit:
- shared dataset, label, metric, and reporting utilities under `offroad_segmentation`,
- config-driven `train.py` and `test.py`,
- checkpoint resume support,
- visual artifact export,
- a FastAPI live inference backend,
- and a React dashboard for run browsing and upload analysis.

### Failure Cases

#### Failure Case 1
![Failure Case 1](Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/worst_cases/worst_000_ww10000002.png)

**What happened:**  
The model struggled in a complex scene with mixed terrain and fine-grained regions.

**Why it likely happened:**  
Rare and small classes receive limited supervision in the current CPU-friendly setup.

**Potential fix:**  
Use class-balanced loss, stronger augmentation, and a richer decoder.

#### Failure Case 2
![Failure Case 2](Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/worst_cases/worst_001_ww10000082.png)

**What happened:**  
Class boundaries were unstable in cluttered terrain.

**Why it likely happened:**  
Low-resolution training and frozen features reduce sensitivity to small details.

**Potential fix:**  
Use higher-resolution training on GPU and partial backbone fine-tuning.

---

## 5. Conclusion & Future Work

### 5.1 Deployment and Demonstration Layer

The project includes a usable demo layer in addition to offline evaluation.

Backend:
- FastAPI server for run discovery, asset serving, and upload inference

Frontend:
- React dashboard for:
  - metric review,
  - split switching,
  - comparison view,
  - fullscreen analysis,
  - live upload inference,
  - and terrain suggestions beside the image

This improves the submission quality because judges can review both:
- quantitative performance,
- and qualitative model behavior on real scenes.

### Conclusion

TerrainScope delivered a complete and reproducible segmentation kit with:
- corrected label handling,
- train and test pipelines,
- validation and test evaluation,
- full artifact export,
- and a live dashboard for browsing runs and uploading images.

The final system is strongest on:
- Sky
- Landscape
- Dry Grass

The main performance gap remains in:
- Logs
- Rocks
- Ground Clutter
- Dry Bushes

Final headline metrics:
- Best Validation Mean IoU: **42.88%**
- Best Validation Mean Dice: **55.42%**
- Best Validation Pixel Accuracy: **78.38%**
- Best Test Mean IoU: **21.21%**
- Best Test Mean Dice: **26.34%**
- Best Test Pixel Accuracy: **61.66%**

### Future Work

1. Train the continuation recipe uncapped on GPU to chase `0.50+` validation IoU.
2. Tune the already-implemented class-balanced loss and sampling to recover test generalization.
3. Fine-tune upper DINOv2 layers instead of freezing the full backbone.
4. Add targeted augmentation for clutter, occlusion, and obstacle-heavy scenes.
5. Compare against a stronger SegFormer run after the new optimizer/loss path.

---

## 6. Reproducibility and Submission Files

### 6.1 Key Output Artifacts

Important generated files for the final run:
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/checkpoints/best_iou.pth`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/plots/training_metrics.png`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/evaluation_metrics.txt`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/per_class_iou.png`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/confusion_matrix.png`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/comparisons/`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/evaluations/val/worst_cases/`

### Reproduction Commands

```powershell
python -m pip install -r .\Offroad_Segmentation_Scripts\requirements.txt
python .\Offroad_Segmentation_Scripts\train.py --run_name dinov2_baseline
python .\Offroad_Segmentation_Scripts\test.py --model_path .\Offroad_Segmentation_Scripts\runs\dinov2_baseline\checkpoints\best_iou.pth --data_root .\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val
python .\Offroad_Segmentation_Scripts\test.py --model_path .\Offroad_Segmentation_Scripts\runs\dinov2_baseline\checkpoints\best_iou.pth --data_root .\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages
```

### Submission Checklist

- `Offroad_Segmentation_Scripts/train.py`
- `Offroad_Segmentation_Scripts/test.py`
- `Offroad_Segmentation_Scripts/configs/`
- `Offroad_Segmentation_Scripts/offroad_segmentation/`
- `Offroad_Segmentation_Scripts/runs/quick_cpu_100x20/`
- `README.md`
- `project_documentation.txt`
- `report.md`
- `final_report.md`

This report is the polished markdown version with percentage-based metrics and embedded charts for export.
