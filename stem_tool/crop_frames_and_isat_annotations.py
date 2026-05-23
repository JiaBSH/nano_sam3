from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

from PIL import Image


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def get_canvas_size_from_mark_dir(mark_dir: Path) -> tuple[int, int]:
    json_files = sorted(mark_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {mark_dir}")

    payload = load_json(json_files[0])
    info = payload.get("info", {})
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in: {json_files[0]}")
    return width, height


def collect_points_from_object(obj: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    segmentation = obj.get("segmentation", [])
    if isinstance(segmentation, list):
        for pt in segmentation:
            if isinstance(pt, list) and len(pt) >= 2:
                x = float(pt[0])
                y = float(pt[1])
                points.append((x, y))

    # Fallback for broken/empty segmentation.
    if not points:
        bbox = obj.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4:
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])
            points.extend([(x1, y1), (x2, y2)])

    return points


def compute_global_crop_box(mark_dir: Path, margin: int) -> tuple[int, int, int, int]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    json_files = sorted(mark_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {mark_dir}")

    width_ref = None
    height_ref = None

    for json_path in json_files:
        payload = load_json(json_path)
        info = payload.get("info", {})
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in: {json_path}")

        if width_ref is None:
            width_ref = width
            height_ref = height
        elif width != width_ref or height != height_ref:
            raise ValueError(
                f"Inconsistent image size in {json_path}: "
                f"({width}, {height}) != ({width_ref}, {height_ref})"
            )

        objects = payload.get("objects", [])
        for obj in objects:
            for x, y in collect_points_from_object(obj):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if not math.isfinite(min_x) or not math.isfinite(min_y):
        raise ValueError(f"No valid annotation points found in: {mark_dir}")

    x1 = max(0, int(math.floor(min_x)) - int(margin))
    y1 = max(0, int(math.floor(min_y)) - int(margin))
    x2 = min(int(width_ref), int(math.ceil(max_x)) + int(margin))
    y2 = min(int(height_ref), int(math.ceil(max_y)) + int(margin))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Computed invalid crop box: ({x1}, {y1}, {x2}, {y2})")

    return x1, y1, x2, y2


def update_object_for_crop(obj: dict, crop_box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    segmentation = obj.get("segmentation", [])
    new_segmentation = []
    if isinstance(segmentation, list):
        for pt in segmentation:
            if isinstance(pt, list) and len(pt) >= 2:
                px = float(pt[0]) - x1
                py = float(pt[1]) - y1
                px = min(max(px, 0.0), float(crop_w))
                py = min(max(py, 0.0), float(crop_h))
                new_segmentation.append([px, py])
            else:
                new_segmentation.append(pt)
        obj["segmentation"] = new_segmentation

    if new_segmentation:
        xs = [pt[0] for pt in new_segmentation if isinstance(pt, list) and len(pt) >= 2]
        ys = [pt[1] for pt in new_segmentation if isinstance(pt, list) and len(pt) >= 2]
        if xs and ys:
            obj["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
            return

    bbox = obj.get("bbox", [])
    if isinstance(bbox, list) and len(bbox) >= 4:
        bx1 = min(max(float(bbox[0]) - x1, 0.0), float(crop_w))
        by1 = min(max(float(bbox[1]) - y1, 0.0), float(crop_h))
        bx2 = min(max(float(bbox[2]) - x1, 0.0), float(crop_w))
        by2 = min(max(float(bbox[3]) - y1, 0.0), float(crop_h))
        obj["bbox"] = [bx1, by1, bx2, by2]


def crop_frames(frame_dir: Path, output_dir: Path, crop_box: tuple[int, int, int, int]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for image_path in sorted(frame_dir.glob("*.jpg")):
        with Image.open(image_path) as image:
            cropped = image.crop(crop_box)
            cropped.save(output_dir / image_path.name)
        count += 1

    return count


def crop_isat_annotations(mark_dir: Path, output_dir: Path, crop_box: tuple[int, int, int, int]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    count = 0

    for json_path in sorted(mark_dir.glob("*.json")):
        payload = load_json(json_path)

        info = payload.get("info", {})
        if isinstance(info, dict):
            info["width"] = crop_w
            info["height"] = crop_h
            payload["info"] = info

        objects = payload.get("objects", [])
        if isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict):
                    update_object_for_crop(obj, crop_box)

        write_json(output_dir / json_path.name, payload)
        count += 1

    isat_yaml = mark_dir / "isat.yaml"
    if isat_yaml.exists():
        copy2(isat_yaml, output_dir / isat_yaml.name)

    return count


def save_crop_metadata(
    meta_path: Path,
    crop_box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    margin: int,
    frame_dir: Path,
    mark_dir: Path,
    output_frame_dir: Path,
    output_mark_dir: Path,
) -> None:
    x1, y1, x2, y2 = crop_box
    source_w, source_h = source_size
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "crop_size": {"width": x2 - x1, "height": y2 - y1},
        "source_size": {"width": source_w, "height": source_h},
        "margin": int(margin),
        # Coordinate transforms for projecting annotations between spaces.
        "transform": {
            "crop_to_source": {"x": "x + x1", "y": "y + y1"},
            "source_to_crop": {"x": "x - x1", "y": "y - y1"},
        },
        "paths": {
            "frame_dir": str(frame_dir),
            "mark_dir": str(mark_dir),
            "output_frame_dir": str(output_frame_dir),
            "output_mark_dir": str(output_mark_dir),
        },
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(meta_path, payload)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_frame_dir = root / "data" / "TEM" / "gas-liquid" / "gas-liquid-frame"
    default_mark_dir = root / "data" / "TEM" / "gas-liquid" / "gas-liquid-mark"
    default_out_frame = root / "data" / "TEM" / "gas-liquid" / "gas-liquid-frame-crop"
    default_out_mark = root / "data" / "TEM" / "gas-liquid" / "gas-liquid-mark-crop"

    parser = argparse.ArgumentParser(
        description="Synchronously crop TEM frames and ISAT annotations using one global ROI from annotations."
    )
    parser.add_argument("--frame-dir", type=Path, default=default_frame_dir, help="Input frame directory.")
    parser.add_argument("--mark-dir", type=Path, default=default_mark_dir, help="Input ISAT JSON directory.")
    parser.add_argument("--output-frame-dir", type=Path, default=default_out_frame, help="Output cropped frame directory.")
    parser.add_argument("--output-mark-dir", type=Path, default=default_out_mark, help="Output cropped ISAT directory.")
    parser.add_argument("--margin", type=int, default=20, help="Extra margin in pixels around annotation union bbox.")
    parser.add_argument(
        "--crop-meta-path",
        type=Path,
        default=None,
        help="Where to save crop parameters JSON (default: <output-mark-dir>/crop_params.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_dir = args.frame_dir.resolve()
    mark_dir = args.mark_dir.resolve()
    output_frame_dir = args.output_frame_dir.resolve()
    output_mark_dir = args.output_mark_dir.resolve()
    crop_meta_path = (
        args.crop_meta_path.resolve() if args.crop_meta_path is not None else (output_mark_dir / "crop_params.json")
    )

    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
    if not mark_dir.is_dir():
        raise FileNotFoundError(f"Mark directory does not exist: {mark_dir}")

    crop_box = compute_global_crop_box(mark_dir=mark_dir, margin=args.margin)
    source_size = get_canvas_size_from_mark_dir(mark_dir)
    frame_count = crop_frames(frame_dir=frame_dir, output_dir=output_frame_dir, crop_box=crop_box)
    mark_count = crop_isat_annotations(mark_dir=mark_dir, output_dir=output_mark_dir, crop_box=crop_box)
    save_crop_metadata(
        meta_path=crop_meta_path,
        crop_box=crop_box,
        source_size=source_size,
        margin=args.margin,
        frame_dir=frame_dir,
        mark_dir=mark_dir,
        output_frame_dir=output_frame_dir,
        output_mark_dir=output_mark_dir,
    )

    x1, y1, x2, y2 = crop_box
    print(f"Crop box (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
    print(f"Crop size: {x2 - x1} x {y2 - y1}")
    print(f"Cropped frames: {frame_count}")
    print(f"Cropped annotation JSON files: {mark_count}")
    print(f"Output frames: {output_frame_dir}")
    print(f"Output ISAT annotations: {output_mark_dir}")
    print(f"Crop params JSON: {crop_meta_path}")


if __name__ == "__main__":
    main()
