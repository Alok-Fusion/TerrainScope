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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from offroad_segmentation.config import config_to_jsonable, load_config
from offroad_segmentation.data import (
    FalconSegmentationDataset,
    compute_class_pixel_counts,
    compute_sample_weights,
    validate_expected_raw_values,
)
from offroad_segmentation.labels import IGNORE_INDEX, NUM_CLASSES
from offroad_segmentation.metrics import create_confusion_matrix, metrics_from_confusion_matrix, update_confusion_matrix
from offroad_segmentation.model import (
    build_segmentation_model,
    checkpoint_metadata_for_model,
    forward_model_logits,
    get_optimization_parameter_groups,
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
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="Initialize model weights from a checkpoint but start a fresh run with the current optimizer/scheduler/config.",
    )
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


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index

        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        present_classes = denominator > 0
        if not torch.any(present_classes):
            return logits.new_tensor(0.0)

        dice_scores = (2.0 * intersection[present_classes] + self.smooth) / (
            denominator[present_classes] + self.smooth
        )
        return 1.0 - dice_scores.mean()


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        *,
        class_weight_tensor: torch.Tensor | None,
        ignore_index: int = IGNORE_INDEX,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if class_weight_tensor is not None:
            self.register_buffer("class_weight_tensor", class_weight_tensor.detach().clone())
        else:
            self.class_weight_tensor = None
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weight_tensor,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        probabilities = torch.softmax(logits, dim=1)
        target_probabilities = probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_factor = torch.pow(1.0 - target_probabilities, self.gamma)
        focal_loss = focal_factor * ce_loss
        return focal_loss[valid_mask].mean()


class CompositeSegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        ce_weight_tensor: torch.Tensor | None,
        ce_weight: float,
        focal_weight: float,
        focal_gamma: float,
        dice_weight: float,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight_tensor, ignore_index=ignore_index)
        self.focal_cross_entropy = FocalCrossEntropyLoss(
            class_weight_tensor=ce_weight_tensor,
            ignore_index=ignore_index,
            gamma=focal_gamma,
        )
        self.dice_loss = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        if self.ce_weight > 0:
            total = total + self.ce_weight * self.cross_entropy(logits, targets)
        if self.focal_weight > 0:
            total = total + self.focal_weight * self.focal_cross_entropy(logits, targets)
        if self.dice_weight > 0:
            total = total + self.dice_weight * self.dice_loss(logits, targets)
        return total


def build_class_weights(config: dict) -> list[float] | None:
    if not config.get("use_class_weights", False):
        return None

    counts = np.asarray(compute_class_pixel_counts(config["train_data_root"], strict_mask=True), dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return None

    frequencies = counts / total
    power = float(config.get("class_weight_power", 0.5))
    epsilon = float(config.get("class_weight_epsilon", 1e-6))
    weights = np.power(frequencies + epsilon, -power)
    max_weight = float(config.get("max_class_weight", 8.0))
    weights = np.clip(weights, 1.0, max_weight)
    weights = weights / weights.mean()
    return weights.astype(np.float32).tolist()


def build_criterion(config: dict, device: torch.device) -> tuple[nn.Module, list[float] | None]:
    class_weights = build_class_weights(config)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    loss_name = str(config.get("loss_name", "cross_entropy")).lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=IGNORE_INDEX), class_weights
    if loss_name in {"cross_entropy_dice", "cross_entropy_focal_dice", "focal_dice"}:
        return (
            CompositeSegmentationLoss(
                num_classes=NUM_CLASSES,
                ce_weight_tensor=weight_tensor,
                ce_weight=0.0 if loss_name == "focal_dice" else float(config.get("ce_loss_weight", 1.0)),
                focal_weight=float(config.get("focal_loss_weight", 0.0))
                if loss_name != "cross_entropy_dice"
                else 0.0,
                focal_gamma=float(config.get("focal_gamma", 2.0)),
                dice_weight=float(config.get("dice_loss_weight", 0.5)),
                ignore_index=IGNORE_INDEX,
            ),
            class_weights,
        )
    raise ValueError(f"Unsupported loss_name '{loss_name}'.")


def build_optimizer(config: dict, parameter_groups: list[dict[str, object]]):
    optimizer_name = str(config.get("optimizer_name", "sgd")).lower()
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        return optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    raise ValueError(f"Unsupported optimizer_name '{optimizer_name}'.")


def build_scheduler(config: dict, optimizer: optim.Optimizer, total_epochs: int):
    scheduler_name = str(config.get("scheduler_name", "none")).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=float(config.get("min_learning_rate", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler_name '{scheduler_name}'.")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
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
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
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
    accumulation_steps: int,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    is_train = mode == "train"
    model.train(is_train)

    confusion_matrix = create_confusion_matrix(num_classes)
    losses: list[float] = []
    progress = tqdm(limited_loader(loader, max_batches), desc=mode.title(), leave=False, unit="batch")

    last_step_index = 0
    for step_index, (images, masks) in enumerate(progress, start=1):
        last_step_index = step_index
        images = images.to(device)
        masks = masks.to(device)

        if optimizer is not None:
            if (step_index - 1) % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = forward_model_logits(model, images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            backward_loss = loss / accumulation_steps

            if is_train and optimizer is not None:
                backward_loss.backward()
                should_step = step_index % accumulation_steps == 0
                if should_step:
                    if gradient_clip_norm is not None and gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
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

    if is_train and optimizer is not None and last_step_index > 0 and last_step_index % accumulation_steps != 0:
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["avg_loss"] = compute_average_loss(losses)
    return metrics


def format_learning_rate_summary(optimizer: optim.Optimizer) -> str:
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not learning_rates:
        return "n/a"
    minimum = min(learning_rates)
    maximum = max(learning_rates)
    if math.isclose(minimum, maximum):
        return f"{minimum:.6f}"
    return f"{minimum:.6f}..{maximum:.6f}"


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
    if resume_checkpoint is not None and not args.resume_weights_only:
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

    if resume_checkpoint_path and not args.resume_weights_only and not args.run_name:
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

    augmentation_config = config.get("augmentation", {})
    train_dataset = FalconSegmentationDataset(
        config["train_data_root"],
        config["image_size"],
        strict_mask=True,
        augment=bool(config.get("train_augment", False)),
        augmentation_config=augmentation_config,
    )
    val_dataset = FalconSegmentationDataset(config["val_data_root"], config["image_size"], strict_mask=True)

    class_weights = build_class_weights(config)
    train_sampler = None
    if config.get("balanced_sampling", False):
        sampler_weights = compute_sample_weights(
            config["train_data_root"],
            class_weights=class_weights or [1.0] * NUM_CLASSES,
            strict_mask=True,
        )
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sampler_weights, dtype=torch.double),
            num_samples=len(sampler_weights),
            replacement=True,
        )

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    if class_weights:
        print(f"Using class-balanced loss weights: {[round(value, 2) for value in class_weights]}")

    model = build_segmentation_model(config, num_classes=NUM_CLASSES, device=device)

    criterion, resolved_class_weights = build_criterion(config, device)
    optimization_parameter_groups = get_optimization_parameter_groups(model, config)
    trainable_parameters = get_trainable_parameters(model)
    if not trainable_parameters or not optimization_parameter_groups:
        raise RuntimeError("No trainable parameters were found. Disable encoder freezing or check model initialization.")

    optimizer = build_optimizer(config, optimization_parameter_groups)
    scheduler = build_scheduler(config, optimizer, int(config["epochs"]))
    accumulation_steps = max(1, int(config.get("accumulation_steps", 1)))
    gradient_clip_norm = float(config.get("gradient_clip_norm", 0.0)) or None

    history = load_history(metrics_dir)
    best_val_iou = -math.inf
    start_epoch = 0

    if resume_checkpoint_path and args.resume_weights_only:
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        load_model_weights(model, resume_checkpoint)
        best_val_iou = infer_best_val_iou(history, None)
        print(f"Initialized model weights from {resume_checkpoint_path} into a fresh run.")
    elif resume_checkpoint_path:
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        load_model_weights(model, resume_checkpoint)
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
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
            accumulation_steps=accumulation_steps,
            gradient_clip_norm=gradient_clip_norm,
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
            accumulation_steps=1,
            gradient_clip_norm=None,
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
            f"val_acc={val_metrics['pixel_accuracy']:.4f} "
            f"lr={format_learning_rate_summary(optimizer)}"
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
                scheduler=scheduler,
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
            scheduler=scheduler,
            config=config,
            epoch=epoch + 1,
            metrics=current_metrics,
            history=history,
            best_val_iou=best_val_iou,
        )

        if scheduler is not None:
            scheduler.step()

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
        "loss_name": str(config.get("loss_name", "cross_entropy")),
        "optimizer_name": str(config.get("optimizer_name", "sgd")),
        "scheduler_name": str(config.get("scheduler_name", "none")),
        "class_weights": resolved_class_weights,
    }
    save_json(summary, metrics_dir / "training_summary.json")

    print(f"\nRun directory: {run_dir}")
    print(f"Best checkpoint: {checkpoint_dir / 'best_iou.pth'}")
    print("Training complete.")


if __name__ == "__main__":
    main()
