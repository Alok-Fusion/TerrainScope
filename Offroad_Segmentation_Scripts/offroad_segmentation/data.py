from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .labels import EXPECTED_RAW_VALUES, NUM_CLASSES, convert_raw_mask_to_class_ids

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


def _load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _apply_color_jitter(image: Image.Image, *, brightness: float, contrast: float, saturation: float) -> Image.Image:
    if brightness > 0:
        factor = 1.0 + np.random.uniform(-brightness, brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if contrast > 0:
        factor = 1.0 + np.random.uniform(-contrast, contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if saturation > 0:
        factor = 1.0 + np.random.uniform(-saturation, saturation)
        image = ImageEnhance.Color(image).enhance(factor)
    return image


def _random_resized_crop(
    image: Image.Image,
    class_mask: np.ndarray,
    output_size: tuple[int, int],
    *,
    scale_range: tuple[float, float],
) -> tuple[Image.Image, np.ndarray]:
    image_width, image_height = image.size
    target_height, target_width = output_size
    target_aspect = target_width / target_height

    scale = float(np.random.uniform(scale_range[0], scale_range[1]))
    crop_width = max(1, int(image_width * scale))
    crop_height = max(1, int(round(crop_width / target_aspect)))

    if crop_height > image_height:
        crop_height = image_height
        crop_width = max(1, int(round(crop_height * target_aspect)))
    crop_width = min(crop_width, image_width)

    max_left = max(0, image_width - crop_width)
    max_top = max(0, image_height - crop_height)
    left = int(np.random.randint(0, max_left + 1)) if max_left > 0 else 0
    top = int(np.random.randint(0, max_top + 1)) if max_top > 0 else 0
    crop_box = (left, top, left + crop_width, top + crop_height)

    image = image.crop(crop_box)
    class_mask = class_mask[top : top + crop_height, left : left + crop_width]
    return image, class_mask


def _preprocess_pair(
    image: Image.Image,
    class_mask: np.ndarray,
    image_size: tuple[int, int],
    *,
    augment: bool,
    augmentation_config: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = augmentation_config or {}
    if augment:
        if float(config.get("horizontal_flip_prob", 0.5)) > np.random.rand():
            image = ImageOps.mirror(image)
            class_mask = np.ascontiguousarray(np.fliplr(class_mask))

        crop_prob = float(config.get("random_crop_prob", 0.8))
        if crop_prob > np.random.rand():
            scale_min = float(config.get("crop_scale_min", 0.75))
            scale_max = float(config.get("crop_scale_max", 1.0))
            image, class_mask = _random_resized_crop(
                image,
                class_mask,
                image_size,
                scale_range=(scale_min, scale_max),
            )

        image = _apply_color_jitter(
            image,
            brightness=float(config.get("brightness_jitter", 0.12)),
            contrast=float(config.get("contrast_jitter", 0.12)),
            saturation=float(config.get("saturation_jitter", 0.08)),
        )

    image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
    image_array = (image_array - IMAGE_MEAN) / IMAGE_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    mask_image = Image.fromarray(class_mask)
    mask_image = mask_image.resize((image_size[1], image_size[0]), Image.Resampling.NEAREST)
    mask_array = np.array(mask_image, dtype=np.uint8, copy=True)
    return torch.from_numpy(image_array).float(), torch.from_numpy(mask_array).long()


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    image = _load_image(image_path)
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
        augment: bool = False,
        augmentation_config: dict | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.image_dir, self.mask_dir = _resolve_split_dirs(self.data_root)
        self.image_size = image_size
        self.strict_mask = strict_mask
        self.return_id = return_id
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.data_ids = _list_image_ids(self.image_dir, self.mask_dir)

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(self, index: int):
        data_id = self.data_ids[index]
        image_path = self.image_dir / data_id
        mask_path = self.mask_dir / data_id

        raw_mask = _load_raw_mask(mask_path)
        class_mask = convert_raw_mask_to_class_ids(raw_mask, strict=self.strict_mask)
        image, mask = _preprocess_pair(
            _load_image(image_path),
            class_mask,
            self.image_size,
            augment=self.augment,
            augmentation_config=self.augmentation_config,
        )

        if self.return_id:
            return image, mask, data_id
        return image, mask


def compute_class_pixel_counts(data_root: str | Path, *, strict_mask: bool = True) -> list[int]:
    _, mask_dir = _resolve_split_dirs(data_root)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for mask_path in sorted(mask_dir.iterdir()):
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        raw_mask = _load_raw_mask(mask_path)
        class_mask = convert_raw_mask_to_class_ids(raw_mask, strict=strict_mask)
        counts += np.bincount(class_mask.reshape(-1), minlength=NUM_CLASSES)

    return counts.tolist()


def compute_sample_weights(
    data_root: str | Path,
    *,
    class_weights: list[float],
    strict_mask: bool = True,
) -> list[float]:
    _, mask_dir = _resolve_split_dirs(data_root)
    weights: list[float] = []
    class_weight_array = np.asarray(class_weights, dtype=np.float32)

    for mask_path in sorted(mask_dir.iterdir()):
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        raw_mask = _load_raw_mask(mask_path)
        class_mask = convert_raw_mask_to_class_ids(raw_mask, strict=strict_mask)
        present_classes = np.unique(class_mask)
        present_classes = present_classes[present_classes < NUM_CLASSES]
        if present_classes.size == 0:
            weights.append(1.0)
            continue
        weights.append(float(class_weight_array[present_classes].max()))

    return weights


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
