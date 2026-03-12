from __future__ import annotations

import numpy as np
import torch


def create_confusion_matrix(num_classes: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    valid = targets != ignore_index

    if valid.sum() == 0:
        return

    predictions = predictions[valid]
    targets = targets[valid]
    encoded = targets * num_classes + predictions
    bincount = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion_matrix += bincount.reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(confusion_matrix: torch.Tensor) -> dict[str, object]:
    matrix = confusion_matrix.detach().cpu().numpy().astype(np.float64)
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive

    union = true_positive + false_positive + false_negative
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)

    dice_denominator = (2.0 * true_positive) + false_positive + false_negative
    dice = np.divide(
        2.0 * true_positive,
        dice_denominator,
        out=np.full_like(true_positive, np.nan),
        where=dice_denominator > 0,
    )

    total = matrix.sum()
    pixel_accuracy = float(true_positive.sum() / total) if total > 0 else float("nan")

    return {
        "mean_iou": float(np.nanmean(iou)),
        "mean_dice": float(np.nanmean(dice)),
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": iou.tolist(),
        "per_class_dice": dice.tolist(),
        "confusion_matrix": matrix.astype(np.int64).tolist(),
    }
