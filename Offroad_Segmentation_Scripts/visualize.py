from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from offroad_segmentation.labels import ensure_class_id_mask, mask_to_color

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colorize raw Falcon masks or predicted class-ID masks.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to a mask file or a directory of masks.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for colorized outputs. Defaults to <input>/colorized.")
    parser.add_argument("--mode", choices=["auto", "raw", "class_id"], default="auto", help="Interpretation mode for input masks.")
    return parser.parse_args()


def iter_mask_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return

    for file_path in sorted(input_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield file_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (input_path.parent if input_path.is_file() else input_path) / "colorized"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for mask_path in iter_mask_files(input_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        class_mask = ensure_class_id_mask(mask, mode=args.mode)
        color_mask = mask_to_color(class_mask)
        output_path = output_dir / f"{mask_path.stem}_color.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        processed += 1

    print(f"Saved {processed} colorized masks to {output_dir}")


if __name__ == "__main__":
    main()
