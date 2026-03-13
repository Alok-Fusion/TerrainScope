from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from offroad_segmentation.config import config_to_jsonable, load_config
from offroad_segmentation.data import FalconSegmentationDataset, validate_raw_values_subset
from offroad_segmentation.labels import IGNORE_INDEX, NUM_CLASSES
from offroad_segmentation.metrics import create_confusion_matrix, metrics_from_confusion_matrix, update_confusion_matrix
from offroad_segmentation.model import build_segmentation_model, forward_model_logits, load_model_weights
from offroad_segmentation.reporting import (
    save_color_mask,
    save_comparison_figure,
    save_confusion_matrix_plot,
    save_evaluation_summary,
    save_json,
    save_per_class_plot,
)

MODEL_CONFIG_KEYS = (
    "model_type",
    "backbone_name",
    "segformer_model_name",
    "deeplab_encoder_name",
    "deeplab_encoder_weights",
    "freeze_encoder",
    "patch_size",
    "image_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model and save predictions.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint created by train.py.")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset split root to evaluate.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory. Defaults to a run-linked eval folder.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of side-by-side comparison images to save.")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit evaluation batches.")
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infer_output_dir(model_path: Path, scripts_dir: Path, data_root: Path) -> Path:
    checkpoint_dir = model_path.resolve().parent
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else scripts_dir / "runs"
    return run_dir / "evaluations" / data_root.name


def limited_loader(loader: DataLoader, max_batches: int | None):
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def per_image_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    confusion_matrix = create_confusion_matrix(NUM_CLASSES)
    update_confusion_matrix(
        confusion_matrix,
        predictions=prediction.cpu(),
        targets=target.cpu(),
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    )
    return float(metrics_from_confusion_matrix(confusion_matrix)["mean_iou"])


def apply_checkpoint_model_config(config: dict, checkpoint: dict | torch.Tensor) -> dict:
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        return config

    checkpoint_config = checkpoint["config"]
    for key in MODEL_CONFIG_KEYS:
        if key in checkpoint_config:
            config[key] = checkpoint_config[key]
    if "image_size" in config:
        config["image_size"] = tuple(int(value) for value in config["image_size"])
    return config


def main() -> None:
    args = parse_args()
    overrides = {
        key: value
        for key, value in {
            "batch_size": args.batch_size,
            "test_data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
            "num_visual_samples": args.num_samples,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides=overrides)
    device = select_device(str(config["device"]))
    print(f"Using device: {device}")

    model_path = Path(args.model_path).resolve() if args.model_path else None
    if model_path is None:
        raise ValueError("`--model_path` is required. Point it to best_iou.pth or last.pth from a run.")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    config = apply_checkpoint_model_config(config, checkpoint)

    data_root = Path(config["test_data_root"]).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else infer_output_dir(model_path, config["scripts_dir"], data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("validate_raw_values", True):
        discovered_values = validate_raw_values_subset([data_root])
        save_json({"raw_values": discovered_values}, output_dir / "data_contract.json")
        print(f"Validated dataset raw values: {discovered_values}")

    dataset = FalconSegmentationDataset(data_root, config["image_size"], strict_mask=True, return_id=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_segmentation_model(config, num_classes=NUM_CLASSES, device=device)
    load_model_weights(model, checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    confusion_matrix = create_confusion_matrix(NUM_CLASSES)
    losses: list[float] = []
    per_image_rows: list[dict[str, float | str]] = []
    sample_limit = int(config["num_visual_samples"])
    worst_case_count = int(config["worst_case_count"])
    sample_records: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
    worst_case_records: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, str]] = []

    raw_mask_dir = output_dir / "predictions" / "raw_masks"
    color_mask_dir = output_dir / "predictions" / "color_masks"
    comparison_dir = output_dir / "comparisons"
    worst_case_dir = output_dir / "worst_cases"

    with torch.no_grad():
        progress = tqdm(limited_loader(loader, args.max_batches), desc="Evaluating", leave=False, unit="batch")
        for images, masks, data_ids in progress:
            images = images.to(device)
            masks = masks.to(device)

            logits = forward_model_logits(model, images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            losses.append(float(loss.item()))

            predictions = torch.argmax(logits, dim=1)
            update_confusion_matrix(
                confusion_matrix,
                predictions=predictions.cpu(),
                targets=masks.cpu(),
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            )

            for index, data_id in enumerate(data_ids):
                predicted_mask = predictions[index].cpu().numpy().astype(np.uint8)
                raw_mask_path = raw_mask_dir / f"{Path(data_id).stem}_pred.png"
                raw_mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(predicted_mask).save(raw_mask_path)
                save_color_mask(predicted_mask, color_mask_dir / f"{Path(data_id).stem}_pred_color.png")

                image_iou = per_image_iou(predictions[index], masks[index])
                per_image_rows.append({"sample_id": data_id, "iou": image_iou})
                record = (
                    image_iou,
                    images[index].detach().cpu(),
                    masks[index].detach().cpu(),
                    predictions[index].detach().cpu(),
                    data_id,
                )
                if len(sample_records) < sample_limit:
                    sample_records.append(record)
                worst_case_records.append(record)
                worst_case_records = sorted(worst_case_records, key=lambda entry: entry[0])[:worst_case_count]

            progress.set_postfix(loss=f"{loss.item():.4f}")

    results = metrics_from_confusion_matrix(confusion_matrix)
    results["avg_loss"] = float(np.mean(losses)) if losses else float("nan")
    results["model_path"] = str(model_path)
    results["data_root"] = str(data_root)
    results["output_dir"] = str(output_dir)
    results["num_images"] = len(per_image_rows)

    save_evaluation_summary(results, output_dir)
    save_per_class_plot(results["per_class_iou"], output_dir / "per_class_iou.png", title="Per-Class IoU", ylabel="IoU")
    save_per_class_plot(results["per_class_dice"], output_dir / "per_class_dice.png", title="Per-Class Dice", ylabel="Dice")
    save_confusion_matrix_plot(results["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_json(config_to_jsonable(config), output_dir / "config_resolved.json")

    with (output_dir / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "iou"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    for sample_index, (_, image, target_mask, prediction, data_id) in enumerate(sample_records[:sample_limit]):
        save_comparison_figure(
            image,
            target_mask,
            prediction,
            comparison_dir / f"sample_{sample_index:03d}_{Path(data_id).stem}.png",
            title=f"Sample {data_id}",
        )

    for worst_index, (sample_iou, image, target_mask, prediction, data_id) in enumerate(worst_case_records):
        save_comparison_figure(
            image,
            target_mask,
            prediction,
            worst_case_dir / f"worst_{worst_index:03d}_{Path(data_id).stem}.png",
            title=f"Worst Case {data_id} | IoU={sample_iou:.4f}",
        )

    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
