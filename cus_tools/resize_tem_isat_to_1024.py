from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = SCRIPT_ROOT / "mmdata_test" / "50x" / "image"
DEFAULT_LABEL_DIR = SCRIPT_ROOT / "mmdata_test" / "50x" / "label"
DEFAULT_OUTPUT_IMAGE_DIR = SCRIPT_ROOT / "mmdata_test_1024" / "50x" / "image"
DEFAULT_OUTPUT_LABEL_DIR = SCRIPT_ROOT / "mmdata_test_1024" / "50x" / "label"
DEFAULT_LONG_SIDE = 1024


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def scale_point(point: list[float], sx: float, sy: float) -> list[float]:
    return [float(point[0]) * sx, float(point[1]) * sy]


def scale_segmentation(segmentation: object, sx: float, sy: float) -> object:
    if not isinstance(segmentation, list):
        return segmentation
    if not segmentation:
        return segmentation

    first = segmentation[0]

    # iSAT polygon style: [[x1, y1], [x2, y2], ...]
    if isinstance(first, list):
        scaled = []
        for point in segmentation:
            if isinstance(point, list) and len(point) >= 2:
                scaled.append(scale_point(point, sx, sy))
            else:
                scaled.append(point)
        return scaled

    # Fallback for flat list style: [x1, y1, x2, y2, ...]
    if isinstance(first, (int, float)) and len(segmentation) % 2 == 0:
        scaled_flat: list[float] = []
        for i in range(0, len(segmentation), 2):
            x = float(segmentation[i]) * sx
            y = float(segmentation[i + 1]) * sy
            scaled_flat.extend([x, y])
        return scaled_flat

    return segmentation


def scale_bbox(bbox: object, sx: float, sy: float) -> object:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return bbox
    return [
        float(bbox[0]) * sx,
        float(bbox[1]) * sy,
        float(bbox[2]) * sx,
        float(bbox[3]) * sy,
    ]


def get_resized_shape(src_w: int, src_h: int, long_side: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source shape: ({src_w}, {src_h})")
    if long_side <= 0:
        raise ValueError(f"Invalid long_side: {long_side}")

    scale = float(long_side) / float(max(src_w, src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    return dst_w, dst_h


def resize_image(image_path: Path, output_path: Path, long_side: int) -> tuple[int, int, int, int]:
    with Image.open(image_path) as image:
        src_w, src_h = image.size
        dst_w, dst_h = get_resized_shape(src_w, src_h, long_side)
        resized = image.resize((dst_w, dst_h), resample=Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path)
    return src_w, src_h, dst_w, dst_h


def process_annotation(
    json_path: Path,
    output_json_path: Path,
    output_image_dir: Path,
    long_side: int,
) -> None:
    payload = load_json(json_path)
    info = payload.get("info", {})

    src_w = int(info.get("width", long_side)) if isinstance(info, dict) else long_side
    src_h = int(info.get("height", long_side)) if isinstance(info, dict) else long_side
    if src_w <= 0 or src_h <= 0:
        src_w = long_side
        src_h = long_side

    target_w, target_h = get_resized_shape(src_w, src_h, long_side)

    sx = float(target_w) / float(src_w)
    sy = float(target_h) / float(src_h)

    if isinstance(info, dict):
        info["width"] = target_w
        info["height"] = target_h
        info["folder"] = str(output_image_dir).replace("\\", "/")
        payload["info"] = info

    objects = payload.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if "segmentation" in obj:
                obj["segmentation"] = scale_segmentation(obj["segmentation"], sx, sy)
            if "bbox" in obj:
                obj["bbox"] = scale_bbox(obj["bbox"], sx, sy)
            if "area" in obj and isinstance(obj["area"], (int, float)):
                obj["area"] = float(obj["area"]) * sx * sy

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize TEM images and synchronized iSAT annotations to target size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Input image directory.")
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="Input iSAT JSON directory.")
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE_DIR,
        help="Output resized image directory.",
    )
    parser.add_argument(
        "--output-label-dir",
        type=Path,
        default=DEFAULT_OUTPUT_LABEL_DIR,
        help="Output resized iSAT JSON directory.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=DEFAULT_LONG_SIDE,
        help="Target long side (aspect ratio is preserved).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = args.image_dir.resolve()
    label_dir = args.label_dir.resolve()
    output_image_dir = args.output_image_dir.resolve()
    output_label_dir = args.output_label_dir.resolve()
    long_side = int(args.long_side)

    if long_side <= 0:
        raise ValueError("Long side must be a positive integer.")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]
    json_paths = sorted(label_dir.glob("*.json"))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {image_dir}")
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {label_dir}")

    resized_images = 0
    resized_labels = 0

    for image_path in image_paths:
        output_path = output_image_dir / image_path.name
        resize_image(image_path, output_path, long_side)
        resized_images += 1

    for json_path in json_paths:
        output_json_path = output_label_dir / json_path.name
        process_annotation(
            json_path=json_path,
            output_json_path=output_json_path,
            output_image_dir=output_image_dir,
            long_side=long_side,
        )
        resized_labels += 1

    isat_yaml = label_dir / "isat.yaml"
    if isat_yaml.exists():
        output_label_dir.mkdir(parents=True, exist_ok=True)
        copy2(isat_yaml, output_label_dir / isat_yaml.name)

    print(f"Input images: {image_dir}")
    print(f"Input labels: {label_dir}")
    print(f"Output images: {output_image_dir}")
    print(f"Output labels: {output_label_dir}")
    print(f"Target long side: {long_side} (aspect ratio preserved)")
    print(f"Resized images: {resized_images}")
    print(f"Resized JSON files: {resized_labels}")


if __name__ == "__main__":
    main()
