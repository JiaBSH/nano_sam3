from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def scale_point(point: list[float], scale: float) -> list[float]:
    return [float(point[0]) * scale, float(point[1]) * scale]


def scale_segmentation(segmentation: object, scale: float) -> object:
    if not isinstance(segmentation, list) or not segmentation:
        return segmentation

    first = segmentation[0]

    # iSAT polygon style: [[x1, y1], [x2, y2], ...]
    if isinstance(first, list):
        scaled: list[object] = []
        for point in segmentation:
            if isinstance(point, list) and len(point) >= 2:
                scaled.append(scale_point(point, scale))
            else:
                scaled.append(point)
        return scaled

    # Fallback for flat list style: [x1, y1, x2, y2, ...]
    if isinstance(first, (int, float)) and len(segmentation) % 2 == 0:
        scaled_flat: list[float] = []
        for i in range(0, len(segmentation), 2):
            x = float(segmentation[i]) * scale
            y = float(segmentation[i + 1]) * scale
            scaled_flat.extend([x, y])
        return scaled_flat

    return segmentation


def scale_bbox(bbox: object, scale: float) -> object:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return bbox
    return [
        float(bbox[0]) * scale,
        float(bbox[1]) * scale,
        float(bbox[2]) * scale,
        float(bbox[3]) * scale,
    ]


def is_isat_annotation(payload: dict) -> bool:
    info = payload.get("info")
    objects = payload.get("objects")
    return isinstance(info, dict) and isinstance(objects, list)


def scale_isat_annotation(payload: dict, scale: float, frame_dir_x2: Path | None) -> dict:
    info = payload.get("info", {})
    if isinstance(info, dict):
        if isinstance(info.get("width"), (int, float)):
            info["width"] = int(round(float(info["width"]) * scale))
        if isinstance(info.get("height"), (int, float)):
            info["height"] = int(round(float(info["height"]) * scale))
        if frame_dir_x2 is not None:
            info["folder"] = str(frame_dir_x2).replace("\\", "/")
        payload["info"] = info

    objects = payload.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if "segmentation" in obj:
                obj["segmentation"] = scale_segmentation(obj["segmentation"], scale)
            if "bbox" in obj:
                obj["bbox"] = scale_bbox(obj["bbox"], scale)
            if isinstance(obj.get("area"), (int, float)):
                obj["area"] = float(obj["area"]) * (scale * scale)

    return payload


def resolve_default_input_dir(root: Path) -> Path:
    mask_dir = root / "data" / "TEM" / "gas-liquid" / "gas-liquid-mask-crop"
    if mask_dir.is_dir():
        return mask_dir
    return root / "data" / "TEM" / "gas-liquid" / "gas-liquid-mark-crop"


def default_output_dir(input_dir: Path, scale: float) -> Path:
    suffix = f"_x{int(scale)}" if float(scale).is_integer() else f"_x{scale}"
    return input_dir.parent / f"{input_dir.name}{suffix}"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_input = resolve_default_input_dir(root)

    parser = argparse.ArgumentParser(
        description="Scale gas-liquid iSAT annotations to match super-resolved frames."
    )
    parser.add_argument(
        "--mark-dir",
        type=Path,
        default=default_input,
        help="Input iSAT annotation directory (JSON).",
    )
    parser.add_argument(
        "--output-mark-dir",
        type=Path,
        default=None,
        help="Output directory for scaled iSAT annotations. Default: <mark-dir>_x2.",
    )
    parser.add_argument(
        "--frame-dir-x2",
        type=Path,
        default=root / "data" / "TEM" / "gas-liquid" / "gas-liquid-frame-crop_x2",
        help="Folder to write into annotation info.folder for x2 frames.",
    )
    parser.add_argument("--scale", type=float, default=2.0, help="Coordinate scaling factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mark_dir = args.mark_dir.resolve()
    if args.output_mark_dir is None:
        output_mark_dir = default_output_dir(mark_dir, args.scale).resolve()
    else:
        output_mark_dir = args.output_mark_dir.resolve()

    frame_dir_x2 = args.frame_dir_x2.resolve() if args.frame_dir_x2 is not None else None

    if args.scale <= 0:
        raise ValueError("--scale must be a positive number.")
    if not mark_dir.is_dir():
        raise FileNotFoundError(f"Input annotation directory does not exist: {mark_dir}")

    json_files = sorted(mark_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {mark_dir}")

    output_mark_dir.mkdir(parents=True, exist_ok=True)

    scaled_count = 0
    copied_json_count = 0

    for json_path in json_files:
        payload = load_json(json_path)
        if is_isat_annotation(payload):
            payload = scale_isat_annotation(payload, scale=args.scale, frame_dir_x2=frame_dir_x2)
            scaled_count += 1
        else:
            copied_json_count += 1
        write_json(output_mark_dir / json_path.name, payload)

    for path in sorted(mark_dir.iterdir()):
        if path.is_file() and path.suffix.lower() != ".json":
            copy2(path, output_mark_dir / path.name)

    print(f"Input annotations: {mark_dir}")
    print(f"Output annotations: {output_mark_dir}")
    print(f"Scale factor: {args.scale}")
    print(f"Scaled iSAT JSON files: {scaled_count}")
    print(f"Copied non-iSAT JSON files: {copied_json_count}")


if __name__ == "__main__":
    main()
