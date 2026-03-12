from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import IMAGE_MEAN, IMAGE_STD
from .labels import CLASS_NAMES, COLOR_PALETTE, mask_to_color

plt.switch_backend("Agg")


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_training_history(history: dict[str, list[float]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_json(history, output_path / "history.json")

    csv_path = output_path / "history.csv"
    fieldnames = list(history.keys())
    row_count = len(history[fieldnames[0]]) if fieldnames else 0

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow({field: history[field][index] for field in fieldnames})


def save_training_plots(history: dict[str, list[float]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_specs = [
        ("Loss", "train_loss", "val_loss"),
        ("IoU", "train_iou", "val_iou"),
        ("Dice", "train_dice", "val_dice"),
        ("Pixel Accuracy", "train_pixel_accuracy", "val_pixel_accuracy"),
    ]

    for axis, (title, train_key, val_key) in zip(axes.flat, plot_specs):
        axis.plot(epochs, history[train_key], label="train", marker="o")
        axis.plot(epochs, history[val_key], label="val", marker="o")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path / "training_metrics.png", dpi=150)
    plt.close(fig)


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGE_STD + IMAGE_MEAN
    image = np.clip(image, 0.0, 1.0)
    return image


def save_comparison_figure(
    image_tensor: torch.Tensor,
    ground_truth_mask: torch.Tensor | np.ndarray,
    predicted_mask: torch.Tensor | np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    image = _denormalize_image(image_tensor)
    ground_truth = ground_truth_mask.detach().cpu().numpy() if isinstance(ground_truth_mask, torch.Tensor) else np.asarray(ground_truth_mask)
    prediction = predicted_mask.detach().cpu().numpy() if isinstance(predicted_mask, torch.Tensor) else np.asarray(predicted_mask)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(mask_to_color(ground_truth))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask_to_color(prediction))
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_per_class_plot(
    values: list[float],
    output_path: str | Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    cleaned_values = [0.0 if np.isnan(value) else value for value in values]
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = [COLOR_PALETTE[index] / 255.0 for index in range(len(CLASS_NAMES))]
    axis.bar(range(len(CLASS_NAMES)), cleaned_values, color=colors, edgecolor="black")
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix_plot(confusion_matrix: list[list[int]], output_path: str | Path) -> None:
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground Truth")
    axis.set_title("Normalized Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", color="black", fontsize=8)

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_summary(results: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_json(results, output_path / "evaluation_metrics.json")

    with (output_path / "evaluation_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("EVALUATION RESULTS\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Mean IoU:        {results['mean_iou']:.4f}\n")
        handle.write(f"Mean Dice:       {results['mean_dice']:.4f}\n")
        handle.write(f"Pixel Accuracy:  {results['pixel_accuracy']:.4f}\n")
        if "avg_loss" in results and results["avg_loss"] == results["avg_loss"]:
            handle.write(f"Average Loss:    {results['avg_loss']:.4f}\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("Per-Class Metrics\n")
        handle.write("-" * 60 + "\n")
        for index, class_name in enumerate(CLASS_NAMES):
            iou = results["per_class_iou"][index]
            dice = results["per_class_dice"][index]
            handle.write(
                f"{class_name:<18} IoU={iou:.4f}  Dice={dice:.4f}\n"
                if not np.isnan(iou) and not np.isnan(dice)
                else f"{class_name:<18} IoU=N/A    Dice=N/A\n"
            )


def save_color_mask(mask: np.ndarray, output_path: str | Path) -> None:
    color_mask = mask_to_color(mask)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
