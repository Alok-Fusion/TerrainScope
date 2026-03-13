from __future__ import annotations

import base64
import csv
import io
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from Offroad_Segmentation_Scripts.offroad_segmentation.data import IMAGE_MEAN, IMAGE_STD
from Offroad_Segmentation_Scripts.offroad_segmentation.labels import CLASS_NAMES, COLOR_PALETTE, NUM_CLASSES, mask_to_color
from Offroad_Segmentation_Scripts.offroad_segmentation.model import (
    build_segmentation_model,
    forward_model_logits,
    load_model_weights,
    model_descriptor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "Offroad_Segmentation_Scripts" / "runs"
CACHE_DIR = REPO_ROOT / "frontend" / ".cache" / "api"

SPLIT_FOLDER_TO_KEY = {
    "val": ("Validation", "val"),
    "Offroad_Segmentation_testImages": ("Test", "test"),
}
ASSET_KINDS = {"image", "prediction", "ground_truth", "comparison"}


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: sanitize_for_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def color_to_hex(color: np.ndarray | list[int]) -> str:
    red, green, blue = [int(channel) for channel in color]
    return f"#{red:02x}{green:02x}{blue:02x}"


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_dir_for_name(run_name: str) -> Path:
    run_dir = (RUNS_DIR / run_name).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_name}")
    return run_dir


def available_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []

    run_dirs = []
    for candidate in RUNS_DIR.iterdir():
        if not candidate.is_dir():
            continue
        if (candidate / "checkpoints" / "best_iou.pth").exists() and (candidate / "metrics" / "history.json").exists():
            run_dirs.append(candidate)
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def classify_split_dir_name(folder_name: str) -> tuple[str, str] | None:
    normalized = folder_name.lower()
    if normalized == "val" or normalized.startswith("val_"):
        return ("Validation", "val")
    if (
        normalized == "offroad_segmentation_testimages"
        or normalized.startswith("test")
        or "test" in normalized
    ):
        return ("Test", "test")
    return None


def preferred_split_dirs(run_dir: Path) -> dict[str, tuple[str, Path]]:
    evaluation_root = run_dir / "evaluations"
    selected: dict[str, tuple[str, Path]] = {}
    if not evaluation_root.exists():
        return selected

    candidates_by_split: dict[str, list[tuple[str, Path]]] = {}
    for candidate in evaluation_root.iterdir():
        if not candidate.is_dir():
            continue
        split_context = classify_split_dir_name(candidate.name)
        if split_context is None:
            continue
        label, split_key = split_context
        if not (candidate / "evaluation_metrics.json").exists():
            continue
        candidates_by_split.setdefault(split_key, []).append((label, candidate))

    for split_key, candidates in candidates_by_split.items():
        candidates.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
        selected[split_key] = candidates[0]

    return selected


def evaluation_dir_for_split(run_dir: Path, split_key: str) -> Path:
    split_dirs = preferred_split_dirs(run_dir)
    split_context = split_dirs.get(split_key)
    if split_context is None:
        raise FileNotFoundError(f"Split '{split_key}' not found for run '{run_dir.name}'.")
    return split_context[1]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    history = read_json(run_dir / "metrics" / "history.json")
    config = read_json(run_dir / "config_resolved.json")
    split_dirs = preferred_split_dirs(run_dir)
    val_dir = split_dirs.get("val", (None, None))[1]
    test_dir = split_dirs.get("test", (None, None))[1]
    val_metrics = read_json(val_dir / "evaluation_metrics.json") if val_dir else None
    test_metrics = read_json(test_dir / "evaluation_metrics.json") if test_dir else None

    return sanitize_for_json(
        {
            "name": run_dir.name,
            "modifiedAt": run_dir.stat().st_mtime,
            "epochs": len(history.get("train_loss", [])),
            "modelType": config.get("model_type", "dinov2"),
            "modelName": model_descriptor(config),
            "backboneName": config.get("backbone_name"),
            "imageSize": config.get("image_size"),
            "bestValIoU": max(history.get("val_iou", [0.0]), default=0.0),
            "valMeanIoU": val_metrics.get("mean_iou") if val_metrics else None,
            "testMeanIoU": test_metrics.get("mean_iou") if test_metrics else None,
        }
    )


def list_runs() -> list[dict[str, Any]]:
    return [summarize_run(run_dir) for run_dir in available_run_dirs()]


def get_default_run_name() -> str:
    run_dirs = available_run_dirs()
    if not run_dirs:
        raise FileNotFoundError("No completed runs were found under Offroad_Segmentation_Scripts/runs.")

    summaries = [summarize_run(run_dir) for run_dir in run_dirs]
    summaries.sort(
        key=lambda run: (
            float(run.get("valMeanIoU") or -1.0),
            float(run.get("bestValIoU") or -1.0),
            float(run.get("modifiedAt") or 0.0),
        ),
        reverse=True,
    )
    return str(summaries[0]["name"])


def parse_metrics_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(path):
        rows.append({"sample_id": row["sample_id"], "iou": float(row["iou"])})
    return rows


def select_featured_samples(rows: list[dict[str, Any]], best_count: int = 3, worst_count: int = 3) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: float(item["iou"]), reverse=True)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    for bucket, candidates in (
        ("Best slice", ordered[:best_count]),
        ("Worst slice", list(reversed(ordered[-worst_count:]))),
    ):
        for candidate in candidates:
            sample_id = str(candidate["sample_id"])
            if sample_id in used:
                continue
            used.add(sample_id)
            selected.append({"sample_id": sample_id, "iou": float(candidate["iou"]), "bucket": bucket})

    return selected


def evaluate_dirs_for_run(run_dir: Path) -> dict[str, tuple[str, Path]]:
    return preferred_split_dirs(run_dir)


def read_image_info(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size

    return {
        "width": width,
        "height": height,
        "fileSizeBytes": image_path.stat().st_size,
        "filename": image_path.name,
    }


def coverage_from_class_mask(class_mask: np.ndarray) -> list[dict[str, Any]]:
    pixel_count = int(class_mask.size)
    class_counts = np.bincount(class_mask.reshape(-1), minlength=NUM_CLASSES)
    coverage = []
    for class_id, pixels in enumerate(class_counts.tolist()):
        ratio = pixels / pixel_count if pixel_count else 0.0
        coverage.append(
            {
                "classId": class_id,
                "className": CLASS_NAMES[class_id],
                "color": color_to_hex(COLOR_PALETTE[class_id]),
                "pixels": pixels,
                "ratio": ratio,
            }
        )
    return sorted(coverage, key=lambda item: item["ratio"], reverse=True)


def build_scene_summary(coverage: list[dict[str, Any]]) -> str:
    if not coverage:
        return "Prediction coverage is unavailable for this image."

    top_classes = [entry for entry in coverage if entry["ratio"] > 0][:3]
    if not top_classes:
        return "The model did not assign a stable class distribution."

    names = [entry["className"] for entry in top_classes]
    if len(names) == 1:
        dominant = names[0]
    elif len(names) == 2:
        dominant = f"{names[0]} and {names[1]}"
    else:
        dominant = f"{names[0]}, {names[1]}, and {names[2]}"

    return f"Scene is dominated by {dominant}."


def build_scene_suggestions(coverage: list[dict[str, Any]], quality_score: float | None) -> list[str]:
    suggestions: list[str] = []
    ratios = {entry["className"]: float(entry["ratio"]) for entry in coverage}

    sky = ratios.get("Sky", 0.0)
    landscape = ratios.get("Landscape", 0.0)
    dry_grass = ratios.get("Dry Grass", 0.0)
    lush_bushes = ratios.get("Lush Bushes", 0.0)
    trees = ratios.get("Trees", 0.0)
    rocks = ratios.get("Rocks", 0.0)
    logs = ratios.get("Logs", 0.0)
    ground_clutter = ratios.get("Ground Clutter", 0.0)
    flowers = ratios.get("Flowers", 0.0)

    obstacle_ratio = rocks + logs + ground_clutter
    vegetation_ratio = dry_grass + lush_bushes + trees

    if sky + landscape >= 0.6:
        suggestions.append("Large open-sky and landscape coverage suggests a broad visible horizon. Use long-range planning, but validate ground-level obstacles locally.")

    if obstacle_ratio >= 0.18:
        suggestions.append("Rocks, logs, or clutter occupy a noticeable part of the frame. Route planning should avoid straight-line traversal through the densest obstacle region.")

    if vegetation_ratio >= 0.45:
        suggestions.append("Vegetation dominates this scene. Expect softer terrain boundaries and watch for segmentation bleed between bushes, grass, and trees.")

    if dry_grass >= 0.18 and lush_bushes >= 0.12:
        suggestions.append("Mixed dry grass and bush coverage indicates edge-heavy terrain. Favor slower turns and double-check the boundary between traversable ground and brush.")

    if flowers >= 0.05:
        suggestions.append("A visible flower patch is present. That usually means fine-grained texture, so small-object boundaries may be less stable than the coarse terrain classes.")

    if quality_score is not None:
        if quality_score < 0.2:
            suggestions.append("Prediction quality is weak on this image. Treat the segmentation as advisory and inspect the comparison view before making decisions.")
        elif quality_score < 0.45:
            suggestions.append("Prediction quality is moderate. Use the dominant classes for context, but verify obstacle and boundary regions manually.")
        else:
            suggestions.append("Prediction quality is relatively strong. The dominant class layout is likely usable for quick scene triage.")

    if not suggestions:
        suggestions.append("Class distribution is balanced without one dominant hazard signature. Inspect the compare view to judge local obstacles and scene boundaries.")

    return suggestions[:4]


def build_analysis_payload(coverage: list[dict[str, Any]], quality_score: float | None, quality_label: str) -> dict[str, Any]:
    return {
        "summary": build_scene_summary(coverage),
        "topClasses": coverage[:3],
        "suggestions": build_scene_suggestions(coverage, quality_score),
        "qualityLabel": quality_label,
        "qualityScore": quality_score,
    }


def sample_asset_url(run_name: str, split_key: str, sample_id: str, asset_kind: str) -> str:
    return f"/api/runs/{run_name}/splits/{split_key}/samples/{sample_id}/{asset_kind}"


def build_dashboard_payload(run_name: str) -> dict[str, Any]:
    run_dir = run_dir_for_name(run_name)
    history = read_json(run_dir / "metrics" / "history.json")
    config = read_json(run_dir / "config_resolved.json")

    payload = {
        "runName": run_name,
        "generatedAt": time.time(),
        "trainingPlot": f"/api/runs/{run_name}/plots/training_metrics.png",
        "history": history,
        "classNames": CLASS_NAMES,
        "classColors": [color_to_hex(color) for color in COLOR_PALETTE.tolist()],
        "config": {
            "device": config.get("device"),
            "modelType": config.get("model_type", "dinov2"),
            "modelName": model_descriptor(config),
            "backboneName": config.get("backbone_name"),
            "imageSize": config.get("image_size"),
            "batchSize": config.get("batch_size"),
            "epochs": config.get("epochs"),
            "patchSize": config.get("patch_size"),
            "learningRate": config.get("learning_rate"),
            "momentum": config.get("momentum"),
        },
        "splits": {},
    }

    for split_key, (label, split_dir) in evaluate_dirs_for_run(run_dir).items():
        metrics = read_json(split_dir / "evaluation_metrics.json")
        rows = parse_metrics_rows(split_dir / "per_image_metrics.csv")
        featured = select_featured_samples(rows)
        samples = []
        source_root = Path(metrics["data_root"])

        for sample in featured:
            sample_id = Path(sample["sample_id"]).stem
            image_path = source_root / "Color_Images" / f"{sample_id}.png"
            prediction_mask_path = split_dir / "predictions" / "raw_masks" / f"{sample_id}_pred.png"
            predicted_mask = np.array(Image.open(prediction_mask_path), dtype=np.uint8)
            coverage = coverage_from_class_mask(predicted_mask)
            samples.append(
                {
                    "id": sample_id,
                    "iou": sample["iou"],
                    "bucket": sample["bucket"],
                    "image": sample_asset_url(run_name, split_key, sample_id, "image"),
                    "prediction": sample_asset_url(run_name, split_key, sample_id, "prediction"),
                    "groundTruth": sample_asset_url(run_name, split_key, sample_id, "ground_truth"),
                    "comparison": sample_asset_url(run_name, split_key, sample_id, "comparison"),
                    "imageInfo": read_image_info(image_path),
                    "analysis": build_analysis_payload(coverage, float(sample["iou"]), "IoU"),
                }
            )

        metrics["num_images"] = int(metrics.get("num_images", len(rows)))
        payload["splits"][split_key] = {
            "label": label,
            "metrics": metrics,
            "samples": samples,
            "plots": {
                "confusionMatrix": f"/api/runs/{run_name}/splits/{split_key}/plots/confusion_matrix.png",
                "perClassIou": f"/api/runs/{run_name}/splits/{split_key}/plots/per_class_iou.png",
            },
        }

    return sanitize_for_json(payload)


def resolve_split_context(run_name: str, split_key: str) -> tuple[Path, dict[str, Any], Path]:
    run_dir = run_dir_for_name(run_name)
    evaluation_dir = evaluation_dir_for_split(run_dir, split_key)
    metrics = read_json(evaluation_dir / "evaluation_metrics.json")
    source_root = Path(metrics["data_root"])
    return run_dir, metrics, source_root


def cache_file_for_sample(run_name: str, split_key: str, sample_id: str, asset_kind: str) -> Path:
    return CACHE_DIR / run_name / split_key / sample_id / f"{asset_kind}.png"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_prediction_asset(source_path: Path, destination: Path, width: int, height: int) -> Path:
    if destination.exists():
        return destination

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Prediction mask not found: {source_path}")

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    ensure_parent(destination)
    cv2.imwrite(str(destination), resized)
    return destination


def ensure_ground_truth_asset(source_path: Path, destination: Path, width: int, height: int) -> Path:
    if destination.exists():
        return destination

    raw_mask = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise FileNotFoundError(f"Ground-truth mask not found: {source_path}")

    if raw_mask.ndim == 3:
        raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)

    from Offroad_Segmentation_Scripts.offroad_segmentation.labels import convert_raw_mask_to_class_ids

    class_mask = convert_raw_mask_to_class_ids(raw_mask, strict=True)
    color_mask = cv2.cvtColor(mask_to_color(class_mask), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    ensure_parent(destination)
    cv2.imwrite(str(destination), resized)
    return destination


def annotate_tile(tile: np.ndarray, label: str) -> np.ndarray:
    output = tile.copy()
    cv2.rectangle(output, (12, 12), (190, 54), (20, 32, 43), thickness=-1)
    cv2.putText(
        output,
        label,
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 248, 235),
        2,
        cv2.LINE_AA,
    )
    return output


def ensure_comparison_asset(image_path: Path, ground_truth_path: Path, prediction_path: Path, destination: Path) -> Path:
    if destination.exists():
        return destination

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_COLOR)
    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_COLOR)
    if image is None or ground_truth is None or prediction is None:
        raise FileNotFoundError(f"Could not create comparison asset for {image_path.stem}")

    tile_height = 300
    scale = tile_height / image.shape[0]
    tile_width = max(1, int(image.shape[1] * scale))
    tiles = []
    for tile, label in ((image, "RGB"), (ground_truth, "GT"), (prediction, "Pred")):
        resized = cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
        tiles.append(annotate_tile(resized, label))

    panel = cv2.hconcat(tiles)
    ensure_parent(destination)
    cv2.imwrite(str(destination), panel)
    return destination


def get_sample_asset_path(run_name: str, split_key: str, sample_id: str, asset_kind: str) -> Path:
    if asset_kind not in ASSET_KINDS:
        raise FileNotFoundError(f"Unknown asset kind: {asset_kind}")

    run_dir, _, source_root = resolve_split_context(run_name, split_key)
    sample_stem = Path(sample_id).stem
    image_path = source_root / "Color_Images" / f"{sample_stem}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if asset_kind == "image":
        return image_path

    evaluation_dir = evaluation_dir_for_split(run_dir, split_key)
    prediction_source = evaluation_dir / "predictions" / "color_masks" / f"{sample_stem}_pred_color.png"
    ground_truth_source = source_root / "Segmentation" / f"{sample_stem}.png"

    image_info = read_image_info(image_path)
    width = int(image_info["width"])
    height = int(image_info["height"])

    if asset_kind == "prediction":
        return ensure_prediction_asset(prediction_source, cache_file_for_sample(run_name, split_key, sample_stem, "prediction"), width, height)

    if asset_kind == "ground_truth":
        return ensure_ground_truth_asset(ground_truth_source, cache_file_for_sample(run_name, split_key, sample_stem, "ground_truth"), width, height)

    prediction_path = get_sample_asset_path(run_name, split_key, sample_stem, "prediction")
    ground_truth_path = get_sample_asset_path(run_name, split_key, sample_stem, "ground_truth")
    comparison_path = cache_file_for_sample(run_name, split_key, sample_stem, "comparison")
    return ensure_comparison_asset(image_path, ground_truth_path, prediction_path, comparison_path)


@lru_cache(maxsize=3)
def load_inference_bundle(run_name: str) -> dict[str, Any]:
    run_dir = run_dir_for_name(run_name)
    config = read_json(run_dir / "config_resolved.json")
    model_path = run_dir / "checkpoints" / "best_iou.pth"
    device = select_device(str(config.get("device", "auto")))

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        for key in (
            "model_type",
            "backbone_name",
            "segformer_model_name",
            "deeplab_encoder_name",
            "deeplab_encoder_weights",
            "freeze_encoder",
            "patch_size",
            "image_size",
        ):
            if key in checkpoint["config"]:
                config[key] = checkpoint["config"][key]
    if "image_size" in config:
        config["image_size"] = tuple(int(value) for value in config["image_size"])

    model = build_segmentation_model(config, num_classes=NUM_CLASSES, device=device)
    load_model_weights(model, checkpoint)
    model.eval()

    return {
        "runName": run_name,
        "runDir": run_dir,
        "config": config,
        "device": device,
        "model": model,
    }


def preprocess_pil_image(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array).float()


def encode_png(image_array_rgb: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode PNG image.")
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def build_overlay_image(image_rgb: np.ndarray, color_mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = (image_rgb.astype(np.float32) * (1.0 - alpha)) + (color_mask_rgb.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_comparison_image(image_rgb: np.ndarray, overlay_rgb: np.ndarray, prediction_rgb: np.ndarray) -> np.ndarray:
    tiles = []
    for tile, label in ((image_rgb, "RGB"), (overlay_rgb, "Overlay"), (prediction_rgb, "Prediction")):
        bgr_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        annotated = annotate_tile(bgr_tile, label)
        tiles.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return np.concatenate(tiles, axis=1)


def infer_uploaded_image(run_name: str, filename: str, file_bytes: bytes) -> dict[str, Any]:
    bundle = load_inference_bundle(run_name)
    config = bundle["config"]
    device = bundle["device"]
    model = bundle["model"]

    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    original_width, original_height = pil_image.size

    start_time = time.perf_counter()
    image_tensor = preprocess_pil_image(pil_image, tuple(int(value) for value in config["image_size"])).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = forward_model_logits(model, image_tensor)
        logits = F.interpolate(logits, size=(original_height, original_width), mode="bilinear", align_corners=False)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)[0].cpu().numpy().astype(np.uint8)
        confidence = probabilities.max(dim=1).values[0].cpu().numpy()

    inference_ms = (time.perf_counter() - start_time) * 1000.0

    original_rgb = np.asarray(pil_image, dtype=np.uint8)
    color_mask_rgb = mask_to_color(prediction)
    overlay_rgb = build_overlay_image(original_rgb, color_mask_rgb)
    comparison_rgb = build_comparison_image(original_rgb, overlay_rgb, color_mask_rgb)

    coverage = coverage_from_class_mask(prediction)
    dominant_class = coverage[0]["className"] if coverage else None

    result = {
        "runName": run_name,
        "modelInfo": {
            "modelType": config.get("model_type", "dinov2"),
            "modelName": model_descriptor(config),
            "backboneName": config.get("backbone_name"),
            "checkpoint": str((bundle["runDir"] / "checkpoints" / "best_iou.pth").resolve()),
            "imageSize": config.get("image_size"),
            "device": str(device),
        },
        "imageInfo": {
            "filename": filename,
            "fileSizeBytes": len(file_bytes),
            "width": original_width,
            "height": original_height,
            "channels": 3,
            "dominantClass": dominant_class,
            "meanConfidence": float(np.mean(confidence)),
            "inferenceMs": inference_ms,
        },
        "classCoverage": coverage,
        "analysis": build_analysis_payload(coverage, float(np.mean(confidence)), "Mean confidence"),
        "images": {
            "original": encode_png(original_rgb),
            "prediction": encode_png(color_mask_rgb),
            "overlay": encode_png(overlay_rgb),
            "comparison": encode_png(comparison_rgb),
        },
    }
    return sanitize_for_json(result)


def plot_path_for_run(run_name: str, plot_name: str, split_key: str | None = None) -> Path:
    run_dir = run_dir_for_name(run_name)
    if split_key is None:
        path = run_dir / "plots" / plot_name
    else:
        path = evaluation_dir_for_split(run_dir, split_key) / plot_name

    if not path.exists():
        raise FileNotFoundError(f"Plot not found: {path}")
    return path
