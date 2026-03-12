from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import EXPECTED_RAW_VALUES, convert_raw_mask_to_class_ids

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _resolve_split_dirs(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root).resolve()
    image_dir = root / "Color_Images"
    mask_dir = root / "Segmentation"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    return image_dir, mask_dir


def _list_image_ids(image_dir: Path, mask_dir: Path) -> list[str]:
    image_files = sorted(
        file.name for file in image_dir.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_files = sorted(
        file.name for file in mask_dir.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if image_files != mask_files:
        image_only = sorted(set(image_files) - set(mask_files))
        mask_only = sorted(set(mask_files) - set(image_files))
        raise ValueError(
            "Color image and mask file lists do not match.\n"
            f"Only in images: {image_only[:5]}\n"
            f"Only in masks: {mask_only[:5]}"
        )

    return image_files


def _load_raw_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
    image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array).float()


def preprocess_mask(mask_path: Path, image_size: tuple[int, int], strict_mask: bool = True) -> torch.Tensor:
    raw_mask = _load_raw_mask(mask_path)
    class_mask = convert_raw_mask_to_class_ids(raw_mask, strict=strict_mask)
    mask_image = Image.fromarray(class_mask)
    mask_image = mask_image.resize((image_size[1], image_size[0]), Image.Resampling.NEAREST)
    mask_array = np.array(mask_image, dtype=np.uint8, copy=True)
    return torch.from_numpy(mask_array).long()


class FalconSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        image_size: tuple[int, int],
        *,
        strict_mask: bool = True,
        return_id: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.image_dir, self.mask_dir = _resolve_split_dirs(self.data_root)
        self.image_size = image_size
        self.strict_mask = strict_mask
        self.return_id = return_id
        self.data_ids = _list_image_ids(self.image_dir, self.mask_dir)

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(self, index: int):
        data_id = self.data_ids[index]
        image_path = self.image_dir / data_id
        mask_path = self.mask_dir / data_id

        image = preprocess_image(image_path, self.image_size)
        mask = preprocess_mask(mask_path, self.image_size, strict_mask=self.strict_mask)

        if self.return_id:
            return image, mask, data_id
        return image, mask


def scan_dataset_raw_values(data_roots: Iterable[str | Path]) -> list[int]:
    discovered_values: set[int] = set()

    for data_root in data_roots:
        _, mask_dir = _resolve_split_dirs(data_root)
        for mask_path in sorted(mask_dir.iterdir()):
            if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            raw_mask = _load_raw_mask(mask_path)
            discovered_values.update(int(value) for value in np.unique(raw_mask).tolist())

    return sorted(discovered_values)


def validate_expected_raw_values(data_roots: Iterable[str | Path]) -> list[int]:
    discovered_values = scan_dataset_raw_values(data_roots)
    expected_values = list(EXPECTED_RAW_VALUES)

    if discovered_values != expected_values:
        raise ValueError(
            "Dataset raw-value contract mismatch. "
            f"Found {discovered_values}, expected {expected_values}."
        )

    return discovered_values


def validate_raw_values_subset(data_roots: Iterable[str | Path]) -> list[int]:
    discovered_values = scan_dataset_raw_values(data_roots)
    expected_value_set = set(EXPECTED_RAW_VALUES)

    if not set(discovered_values).issubset(expected_value_set):
        raise ValueError(
            "Dataset contains raw values outside the supported contract. "
            f"Found {discovered_values}, expected subset of {list(EXPECTED_RAW_VALUES)}."
        )

    return discovered_values
