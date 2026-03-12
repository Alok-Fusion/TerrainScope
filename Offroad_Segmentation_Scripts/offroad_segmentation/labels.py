from __future__ import annotations

from typing import Iterable

import numpy as np

IGNORE_INDEX = 255

RAW_TO_CLASS_ID = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9,
}

CLASS_ID_TO_RAW = {class_id: raw_value for raw_value, class_id in RAW_TO_CLASS_ID.items()}

CLASS_NAMES = [
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

COLOR_PALETTE = np.array(
    [
        [34, 139, 34],
        [0, 196, 82],
        [210, 180, 140],
        [139, 90, 43],
        [128, 128, 0],
        [255, 105, 180],
        [139, 69, 19],
        [128, 128, 128],
        [160, 82, 45],
        [135, 206, 235],
    ],
    dtype=np.uint8,
)

EXPECTED_RAW_VALUES = tuple(sorted(RAW_TO_CLASS_ID.keys()))
NUM_CLASSES = len(CLASS_NAMES)


def _sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted(int(value) for value in values)


def convert_raw_mask_to_class_ids(mask: np.ndarray, strict: bool = True) -> np.ndarray:
    raw_mask = np.asarray(mask)
    class_mask = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.uint8)

    for raw_value, class_id in RAW_TO_CLASS_ID.items():
        class_mask[raw_mask == raw_value] = class_id

    unknown_values = _sorted_ints(np.unique(raw_mask[class_mask == IGNORE_INDEX]))
    if strict and unknown_values:
        raise ValueError(
            "Encountered unknown mask values: "
            f"{unknown_values}. Expected exactly {list(EXPECTED_RAW_VALUES)}."
        )

    return class_mask


def detect_mask_mode(mask: np.ndarray) -> str:
    values = _sorted_ints(np.unique(mask))
    raw_value_set = set(EXPECTED_RAW_VALUES)
    class_value_set = set(range(NUM_CLASSES))

    if set(values).issubset(raw_value_set):
        return "raw"
    if set(values).issubset(class_value_set | {IGNORE_INDEX}):
        return "class_id"
    return "unknown"


def ensure_class_id_mask(mask: np.ndarray, mode: str = "auto") -> np.ndarray:
    detected_mode = detect_mask_mode(mask) if mode == "auto" else mode

    if detected_mode == "raw":
        return convert_raw_mask_to_class_ids(mask, strict=True)
    if detected_mode == "class_id":
        class_mask = np.asarray(mask, dtype=np.uint8)
        unknown_values = _sorted_ints(
            np.unique(class_mask[(class_mask != IGNORE_INDEX) & (class_mask >= NUM_CLASSES)])
        )
        if unknown_values:
            raise ValueError(
                f"Class-ID mask contains values outside 0..{NUM_CLASSES - 1}: {unknown_values}"
            )
        return class_mask

    raise ValueError(
        "Could not detect mask mode. Expected raw Falcon values "
        f"{list(EXPECTED_RAW_VALUES)} or class IDs 0..{NUM_CLASSES - 1}."
    )


def mask_to_color(mask: np.ndarray, ignore_color: tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
    class_mask = np.asarray(mask)
    color_mask = np.zeros((*class_mask.shape, 3), dtype=np.uint8)

    for class_id, rgb in enumerate(COLOR_PALETTE):
        color_mask[class_mask == class_id] = rgb

    color_mask[class_mask == IGNORE_INDEX] = np.asarray(ignore_color, dtype=np.uint8)
    return color_mask
