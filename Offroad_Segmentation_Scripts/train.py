from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from offroad_segmentation.config import config_to_jsonable, load_config
from offroad_segmentation.data import FalconSegmentationDataset, validate_expected_raw_values
from offroad_segmentation.labels import IGNORE_INDEX, NUM_CLASSES
from offroad_segmentation.metrics import create_confusion_matrix, metrics_from_confusion_matrix, update_confusion_matrix
from offroad_segmentation.model import (
    build_segmentation_model,
    checkpoint_metadata_for_model,
    forward_model_logits,
    get_trainable_parameters,
    load_model_weights,
    model_descriptor,
)
from offroad_segmentation.reporting import save_json, save_training_history, save_training_plots

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
    parser = argparse.ArgumentParser(description="Train the off-road segmentation models with CPU-safe defaults.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--data_root", type=str, default=None, help="Override training dataset root.")
    parser.add_argument("--val_root", type=str, default=None, help="Override validation dataset root.")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the run directory.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a last.pth/best_iou.pth checkpoint to continue from.")
    parser.add_argument("--model_type", type=str, default=None, help="Model type: dinov2, segformer_b0, or deeplabv3plus.")
    parser.add_argument("--backbone_name", type=str, default=None, help="Override DINOv2 backbone name, such as dinov2_vitb14.")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Limit train batches per epoch.")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Limit validation batches per epoch.")
    parser.add_argument("--dry_run", action="store_true", help="Run a single short epoch and still emit artifacts.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_run_dir(scripts_dir: Path, requested_name: str) -> Path:
    run_name = requested_name or "run"
    run_dir = scripts_dir / "runs" / run_name
    if run_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = scripts_dir / "runs" / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_dir = checkpoint_path.resolve().parent
    if checkpoint_dir.name == "checkpoints":
        return checkpoint_dir.parent
    return checkpoint_dir


def limited_loader(loader: DataLoader, max_batches: int | None):
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


def compute_average_loss(losses: list[float]) -> float:
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    config: dict,
    epoch: int,
    metrics: dict[str, float],
    history: dict[str, list[float]],
    best_val_iou: float,
) -> None:
    payload = {
        **checkpoint_metadata_for_model(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_to_jsonable(config),
        "epoch": epoch,
        "metrics": metrics,
        "history": history,
        "best_val_iou": best_val_iou,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_history(metrics_dir: Path) -> dict[str, list[float]]:
    history_path = metrics_dir / "history.json"
    if not history_path.exists():
        return {
            "train_loss": [],
            "val_loss": [],
            "train_iou": [],
            "val_iou": [],
            "train_dice": [],
            "val_dice": [],
            "train_pixel_accuracy": [],
            "val_pixel_accuracy": [],
        }

    with history_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_best_val_iou(history: dict[str, list[float]], checkpoint: dict | None) -> float:
    history_values = history.get("val_iou", [])
    if history_values:
        return float(max(history_values))
    if checkpoint and "best_val_iou" in checkpoint:
        return float(checkpoint["best_val_iou"])
    if checkpoint and "metrics" in checkpoint and "val_iou" in checkpoint["metrics"]:
        return float(checkpoint["metrics"]["val_iou"])
    return -math.inf


def apply_checkpoint_model_config(config: dict, checkpoint: dict | None) -> dict:
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        return config
    checkpoint_config = checkpoint["config"]
    for key in MODEL_CONFIG_KEYS:
        if key in checkpoint_config:
            config[key] = checkpoint_config[key]
    if "image_size" in config:
        config["image_size"] = tuple(int(value) for value in config["image_size"])
    return config


def run_epoch(
    *,
    mode: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    num_classes: int,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = mode == "train"
    model.train(is_train)

    confusion_matrix = create_confusion_matrix(num_classes)
    losses: list[float] = []
    progress = tqdm(limited_loader(loader, max_batches), desc=mode.title(), leave=False, unit="batch")

    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = forward_model_logits(model, images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)

            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits, dim=1).detach().cpu()
        update_confusion_matrix(
            confusion_matrix,
            predictions=predictions,
            targets=masks.detach().cpu(),
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
        )
        losses.append(float(loss.item()))
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["avg_loss"] = compute_average_loss(losses)
    return metrics


def main() -> None:
    args = parse_args()
    resume_checkpoint_path = Path(args.resume_from).resolve() if args.resume_from else None
    resume_checkpoint = None
    if resume_checkpoint_path:
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
            "val_data_root": str(Path(args.val_root).resolve()) if args.val_root else None,
            "run_name": args.run_name if args.run_name else None,
            "model_type": args.model_type if args.model_type else None,
            "backbone_name": args.backbone_name if args.backbone_name else None,
        }.items()
        if value is not None
    }

    config = load_config(args.config, overrides=overrides)
    config = apply_checkpoint_model_config(config, resume_checkpoint)
    if args.dry_run:
        config["epochs"] = 1

    seed_everything(int(config["seed"]))
    device = select_device(str(config["device"]))
    print(f"Using device: {device}")
    print(
        "Config summary: "
        f"model_type={config['model_type']}, "
        f"model={model_descriptor(config)}, "
        f"image_size={config['image_size'][0]}x{config['image_size'][1]}, "
        f"batch_size={config['batch_size']}, "
        f"epochs={config['epochs']}"
    )

    if resume_checkpoint_path:
        run_dir = infer_run_dir_from_checkpoint(resume_checkpoint_path)
    else:
        run_dir = build_run_dir(config["scripts_dir"], str(config["run_name"]))
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    metrics_dir = run_dir / "metrics"

    save_json(config_to_jsonable(config), run_dir / "config_resolved.json")

    if config.get("validate_raw_values", True):
        discovered_values = validate_expected_raw_values([config["train_data_root"], config["val_data_root"]])
        save_json({"raw_values": discovered_values}, run_dir / "data_contract.json")
        print(f"Validated dataset raw values: {discovered_values}")

    train_dataset = FalconSegmentationDataset(config["train_data_root"], config["image_size"], strict_mask=True)
    val_dataset = FalconSegmentationDataset(config["val_data_root"], config["image_size"], strict_mask=True)

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = build_segmentation_model(config, num_classes=NUM_CLASSES, device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    trainable_parameters = get_trainable_parameters(model)
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters were found. Disable encoder freezing or check model initialization.")

    optimizer = optim.SGD(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = load_history(metrics_dir)
    best_val_iou = -math.inf
    start_epoch = 0

    if resume_checkpoint_path:
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        load_model_weights(model, resume_checkpoint)
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_val_iou = infer_best_val_iou(history, resume_checkpoint)
        print(f"Resuming from epoch {start_epoch} using {resume_checkpoint_path}")
    else:
        best_val_iou = infer_best_val_iou(history, None)

    epochs_to_run = int(config["epochs"])
    end_epoch = start_epoch + epochs_to_run

    for epoch in range(start_epoch, end_epoch):
        print(f"\nEpoch {epoch + 1}/{end_epoch}")
        train_metrics = run_epoch(
            mode="train",
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            num_classes=NUM_CLASSES,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            mode="val",
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            num_classes=NUM_CLASSES,
            max_batches=args.max_val_batches,
        )

        history["train_loss"].append(train_metrics["avg_loss"])
        history["val_loss"].append(val_metrics["avg_loss"])
        history["train_iou"].append(train_metrics["mean_iou"])
        history["val_iou"].append(val_metrics["mean_iou"])
        history["train_dice"].append(train_metrics["mean_dice"])
        history["val_dice"].append(val_metrics["mean_dice"])
        history["train_pixel_accuracy"].append(train_metrics["pixel_accuracy"])
        history["val_pixel_accuracy"].append(val_metrics["pixel_accuracy"])

        print(
            f"train_loss={train_metrics['avg_loss']:.4f} "
            f"val_loss={val_metrics['avg_loss']:.4f} "
            f"val_iou={val_metrics['mean_iou']:.4f} "
            f"val_dice={val_metrics['mean_dice']:.4f} "
            f"val_acc={val_metrics['pixel_accuracy']:.4f}"
        )

        save_training_history(history, metrics_dir)
        save_training_plots(history, plots_dir)

        current_metrics = {
            "val_iou": float(val_metrics["mean_iou"]),
            "val_dice": float(val_metrics["mean_dice"]),
            "val_accuracy": float(val_metrics["pixel_accuracy"]),
            "val_loss": float(val_metrics["avg_loss"]),
        }

        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = float(val_metrics["mean_iou"])
            save_checkpoint(
                checkpoint_dir / "best_iou.pth",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch + 1,
                metrics=current_metrics,
                history=history,
                best_val_iou=best_val_iou,
            )

        save_checkpoint(
            checkpoint_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch + 1,
            metrics=current_metrics,
            history=history,
            best_val_iou=best_val_iou,
        )

    summary = {
        "best_val_iou": best_val_iou,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_iou": history["train_iou"][-1],
        "final_val_iou": history["val_iou"][-1],
        "final_train_dice": history["train_dice"][-1],
        "final_val_dice": history["val_dice"][-1],
        "final_train_pixel_accuracy": history["train_pixel_accuracy"][-1],
        "final_val_pixel_accuracy": history["val_pixel_accuracy"][-1],
    }
    save_json(summary, metrics_dir / "training_summary.json")

    print(f"\nRun directory: {run_dir}")
    print(f"Best checkpoint: {checkpoint_dir / 'best_iou.pth'}")
    print("Training complete.")


if __name__ == "__main__":
    main()
