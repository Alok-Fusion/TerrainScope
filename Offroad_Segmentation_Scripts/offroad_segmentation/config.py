from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "default_config.json"
PATCH_SIZE_REQUIRED_MODELS = {"dinov2"}
FIXED_DIVISIBILITY = {"deeplabv3plus": 16}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def config_to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [config_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [config_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: config_to_jsonable(item) for key, item in value.items()}
    return value


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base_config = _load_json(DEFAULT_CONFIG_PATH)
    active_config_path = DEFAULT_CONFIG_PATH

    if config_path:
        active_config_path = Path(config_path).resolve()
        user_config = _load_json(active_config_path)
        base_config = _deep_update(base_config, user_config)

    if overrides:
        base_config = _deep_update(base_config, overrides)

    config_dir = active_config_path.parent
    base_config["config_path"] = active_config_path
    base_config["scripts_dir"] = SCRIPT_DIR
    base_config["model_type"] = str(base_config.get("model_type", "dinov2")).lower()
    base_config["image_size"] = tuple(int(value) for value in base_config["image_size"])
    base_config["freeze_encoder"] = bool(base_config.get("freeze_encoder", True))

    if base_config["model_type"] in PATCH_SIZE_REQUIRED_MODELS:
        patch_size = int(base_config["patch_size"])
        image_height, image_width = base_config["image_size"]
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError(
                f"Image size {base_config['image_size']} must be divisible by patch size {patch_size}."
            )
    if base_config["model_type"] in FIXED_DIVISIBILITY:
        divisor = FIXED_DIVISIBILITY[base_config["model_type"]]
        image_height, image_width = base_config["image_size"]
        if image_height % divisor != 0 or image_width % divisor != 0:
            raise ValueError(
                f"Image size {base_config['image_size']} must be divisible by {divisor} for {base_config['model_type']}."
            )

    for key in ("train_data_root", "val_data_root", "test_data_root"):
        base_config[key] = _resolve_path(base_config.get(key), config_dir)

    return base_config
