from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "Offroad_Segmentation_Scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from offroad_segmentation.labels import CLASS_NAMES, COLOR_PALETTE, convert_raw_mask_to_class_ids, mask_to_color


SPLIT_FOLDERS = {
    "val": ("Validation", "val"),
    "Offroad_Segmentation_testImages": ("Test", "test"),
}


@dataclass
class SampleRecord:
    sample_id: str
    iou: float
    bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export run artifacts for the React dashboard.")
    parser.add_argument(
        "--run-dir",
        default=REPO_ROOT / "Offroad_Segmentation_Scripts" / "runs" / "quick_cpu_100x20",
        type=Path,
        help="Training run directory to export from.",
    )
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "frontend" / "public" / "dashboard",
        type=Path,
        help="Dashboard asset output directory.",
    )
    parser.add_argument(
        "--best-count",
        default=3,
        type=int,
        help="Number of highest-IoU samples per split.",
    )
    parser.add_argument(
        "--worst-count",
        default=3,
        type=int,
        help="Number of lowest-IoU samples per split.",
    )
    return parser.parse_args()


def to_hex(rgb_triplet: list[int]) -> str:
    return "#%02x%02x%02x" % tuple(rgb_triplet)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_per_image_metrics(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({"sample_id": row["sample_id"], "iou": float(row["iou"])})
        return rows


def select_samples(rows: list[dict[str, float | str]], best_count: int, worst_count: int) -> list[SampleRecord]:
    ordered = sorted(rows, key=lambda item: float(item["iou"]), reverse=True)
    selected: list[SampleRecord] = []
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
            selected.append(
                SampleRecord(
                    sample_id=sample_id,
                    iou=float(candidate["iou"]),
                    bucket=bucket,
                )
            )

    return selected


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sanitize_for_json(value):
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {key: sanitize_for_json(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def resize_image(source: Path, destination: Path, width: int, height: int) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    interpolation = cv2.INTER_NEAREST if source.name.endswith("_pred_color.png") else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    cv2.imwrite(str(destination), resized)


def write_ground_truth_mask(source_mask: Path, destination: Path, width: int, height: int) -> None:
    raw_mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise FileNotFoundError(f"Could not read mask: {source_mask}")

    class_mask = convert_raw_mask_to_class_ids(raw_mask, strict=True)
    rgb_mask = mask_to_color(class_mask)
    bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(destination), resized)


def annotate_tile(tile, label: str):
    output = tile.copy()
    cv2.rectangle(output, (12, 12), (170, 54), (20, 32, 43), thickness=-1)
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


def write_comparison_panel(image_path: Path, ground_truth_path: Path, prediction_path: Path, destination: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_COLOR)
    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_COLOR)
    if image is None or ground_truth is None or prediction is None:
        raise FileNotFoundError(f"Could not build comparison panel for {image_path.stem}")

    tile_height = 280
    scale = tile_height / image.shape[0]
    tile_width = int(image.shape[1] * scale)

    tiles = []
    for tile, label in ((image, "RGB"), (ground_truth, "GT"), (prediction, "Pred")):
        resized = cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
        tiles.append(annotate_tile(resized, label))

    panel = cv2.hconcat(tiles)
    cv2.imwrite(str(destination), panel)


def export_split(
    split_folder_name: str,
    split_label: str,
    split_key: str,
    run_dir: Path,
    output_dir: Path,
    best_count: int,
    worst_count: int,
) -> dict:
    evaluation_dir = run_dir / "evaluations" / split_folder_name
    metrics = load_json(evaluation_dir / "evaluation_metrics.json")
    rows = load_per_image_metrics(evaluation_dir / "per_image_metrics.csv")
    featured_samples = select_samples(rows, best_count=best_count, worst_count=worst_count)

    source_root = Path(metrics["data_root"])
    split_output_dir = output_dir / split_key
    split_output_dir.mkdir(parents=True, exist_ok=True)

    copy_file(evaluation_dir / "confusion_matrix.png", split_output_dir / "confusion_matrix.png")
    copy_file(evaluation_dir / "per_class_iou.png", split_output_dir / "per_class_iou.png")

    serialized_samples = []
    for sample in featured_samples:
        stem = Path(sample.sample_id).stem
        sample_dir = split_output_dir / "samples" / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        original_image = source_root / "Color_Images" / f"{stem}.png"
        ground_truth_raw = source_root / "Segmentation" / f"{stem}.png"
        prediction_color = evaluation_dir / "predictions" / "color_masks" / f"{stem}_pred_color.png"

        copy_file(original_image, sample_dir / "image.png")
        original_bgr = cv2.imread(str(original_image), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise FileNotFoundError(f"Could not read original image: {original_image}")

        height, width = original_bgr.shape[:2]
        resize_image(prediction_color, sample_dir / "prediction.png", width=width, height=height)
        write_ground_truth_mask(ground_truth_raw, sample_dir / "ground_truth.png", width=width, height=height)
        write_comparison_panel(
            sample_dir / "image.png",
            sample_dir / "ground_truth.png",
            sample_dir / "prediction.png",
            sample_dir / "comparison.png",
        )

        serialized_samples.append(
            {
                "id": stem,
                "iou": sample.iou,
                "bucket": sample.bucket,
                "image": f"/dashboard/{split_key}/samples/{stem}/image.png",
                "prediction": f"/dashboard/{split_key}/samples/{stem}/prediction.png",
                "groundTruth": f"/dashboard/{split_key}/samples/{stem}/ground_truth.png",
                "comparison": f"/dashboard/{split_key}/samples/{stem}/comparison.png",
            }
        )

    metrics["num_images"] = int(metrics.get("num_images", len(rows)))
    return {
        "label": split_label,
        "metrics": metrics,
        "samples": serialized_samples,
        "plots": {
            "confusionMatrix": f"/dashboard/{split_key}/confusion_matrix.png",
            "perClassIou": f"/dashboard/{split_key}/per_class_iou.png",
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()

    ensure_clean_dir(output_dir)

    history = load_json(run_dir / "metrics" / "history.json")
    config = load_json(run_dir / "config_resolved.json")
    copy_file(run_dir / "plots" / "training_metrics.png", output_dir / "training_metrics.png")

    payload = {
        "runName": run_dir.name,
        "generatedAt": run_dir.stat().st_mtime,
        "trainingPlot": "/dashboard/training_metrics.png",
        "history": history,
        "classNames": CLASS_NAMES,
        "classColors": [to_hex(list(color)) for color in COLOR_PALETTE.tolist()],
        "config": {
            "device": config["device"],
            "backboneName": config["backbone_name"],
            "imageSize": config["image_size"],
            "batchSize": config["batch_size"],
            "epochs": config["epochs"],
            "patchSize": config["patch_size"],
            "learningRate": config["learning_rate"],
            "momentum": config["momentum"],
        },
        "splits": {},
    }

    for folder_name, (label, split_key) in SPLIT_FOLDERS.items():
        payload["splits"][split_key] = export_split(
            split_folder_name=folder_name,
            split_label=label,
            split_key=split_key,
            run_dir=run_dir,
            output_dir=output_dir,
            best_count=args.best_count,
            worst_count=args.worst_count,
        )

    with (output_dir / "data.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, indent=2)

    print(f"Exported dashboard assets to {output_dir}")


if __name__ == "__main__":
    main()
