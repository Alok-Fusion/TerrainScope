from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .service import (
    build_dashboard_payload,
    get_default_run_name,
    get_sample_asset_path,
    infer_uploaded_image,
    list_runs,
    plot_path_for_run,
)

app = FastAPI(title="TerrainScope API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def runs() -> dict[str, object]:
    run_list = list_runs()
    return {"runs": run_list, "defaultRun": run_list[0]["name"] if run_list else None}


@app.get("/api/dashboard")
def dashboard(run_name: str | None = None) -> dict[str, object]:
    active_run = run_name or get_default_run_name()
    try:
        return build_dashboard_payload(active_run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_name}/plots/{plot_name}")
def run_plot(run_name: str, plot_name: str):
    try:
        return FileResponse(plot_path_for_run(run_name, plot_name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_name}/splits/{split_key}/plots/{plot_name}")
def split_plot(run_name: str, split_key: str, plot_name: str):
    try:
        return FileResponse(plot_path_for_run(run_name, plot_name, split_key=split_key))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_name}/splits/{split_key}/samples/{sample_id}/{asset_kind}")
def sample_asset(run_name: str, split_key: str, sample_id: str, asset_kind: str):
    try:
        return FileResponse(get_sample_asset_path(run_name, split_key, sample_id, asset_kind))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/inference")
async def inference(run_name: str | None = None, file: UploadFile = File(...)) -> dict[str, object]:
    active_run = run_name or get_default_run_name()
    try:
        payload = infer_uploaded_image(active_run, file.filename or "uploaded-image.png", await file.read())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return payload
