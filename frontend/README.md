# TerrainScope Frontend

Styled React dashboard with a local FastAPI backend for:

- run switching without manual export
- image upload and backend inference
- fullscreen compare modal
- live sample assets pulled from the selected run

## Local setup

From the repo root:

```powershell
\.\.venv\Scripts\uvicorn frontend.server.app:app --reload --port 8000
```

In a second terminal:

```powershell
cd .\frontend
npm install
npm run dev
```

Open the local URL shown by Vite. The frontend talks to `http://127.0.0.1:8000` by default.

## API features

- `GET /api/runs`: list available runs
- `GET /api/dashboard?run_name=<run>`: load metrics and gallery data for a run
- `POST /api/inference?run_name=<run>`: upload one image and receive prediction assets plus image metadata

## What the UI shows

- validation and test metrics
- training curves
- per-class IoU and Dice
- confusion heatmap
- live slider for original image vs prediction or ground truth
- upload inference with filename, size, dimensions, confidence, inference time, and class coverage
- curated best/worst sample gallery
- fullscreen compare modal

## Static export mode

The old export script still exists if you want a frozen dashboard snapshot:

```powershell
python .\frontend\scripts\export_dashboard_assets.py --run-dir .\Offroad_Segmentation_Scripts\runs\<run_name>
```
