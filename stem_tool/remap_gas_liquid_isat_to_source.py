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


def is_isat_annotation(payload: dict) -> bool:
    return isinstance(payload.get("info"), dict) and isinstance(payload.get("objects"), list)


def transform_point(point: list[float], scale: float, offset_x: float, offset_y: float) -> list[float]:
    return [float(point[0]) * scale + offset_x, float(point[1]) * scale + offset_y]


def transform_segmentation(segmentation: object, scale: float, offset_x: float, offset_y: float) -> object:
    if not isinstance(segmentation, list) or not segmentation:
        return segmentation

    first = segmentation[0]

    # iSAT polygon style: [[x1, y1], [x2, y2], ...]
    if isinstance(first, list):
        remapped: list[object] = []
        for point in segmentation:
            if isinstance(point, list) and len(point) >= 2:
                remapped.append(transform_point(point, scale, offset_x, offset_y))
            else:
                remapped.append(point)
        return remapped

    # Fallback for flat list style: [x1, y1, x2, y2, ...]
    if isinstance(first, (int, float)) and len(segmentation) % 2 == 0:
        remapped_flat: list[float] = []
        for index in range(0, len(segmentation), 2):
            x = float(segmentation[index]) * scale + offset_x
            y = float(segmentation[index + 1]) * scale + offset_y
            remapped_flat.extend([x, y])
        return remapped_flat

    return segmentation


def transform_bbox(bbox: object, scale: float, offset_x: float, offset_y: float) -> object:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return bbox
    return [
        float(bbox[0]) * scale + offset_x,
        float(bbox[1]) * scale + offset_y,
        float(bbox[2]) * scale + offset_x,
        float(bbox[3]) * scale + offset_y,
    ]


def remap_isat_annotation(
    payload: dict,
    scale: float,
    offset_x: float,
    offset_y: float,
    source_width: int,
    source_height: int,
    source_folder: str | None,
) -> dict:
    info = payload.get("info", {})
    if isinstance(info, dict):
        info["width"] = int(source_width)
        info["height"] = int(source_height)
        if source_folder:
            info["folder"] = source_folder
        payload["info"] = info

    objects = payload.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if "segmentation" in obj:
                obj["segmentation"] = transform_segmentation(obj["segmentation"], scale, offset_x, offset_y)
            if "bbox" in obj:
                obj["bbox"] = transform_bbox(obj["bbox"], scale, offset_x, offset_y)
            if isinstance(obj.get("area"), (int, float)):
                obj["area"] = float(obj["area"]) * (scale * scale)

    return payload


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_input_dir = repo_root / "outputs" / "gas-liquid-first-frame-sam3" / "mask"
    default_crop_params = repo_root / "data_cus" / "gas-liquid-frame" / "crop_params.json"
    default_output_dir = repo_root / "outputs" / "gas-liquid-first-frame-sam3" / "mask_source"

    parser = argparse.ArgumentParser(
        description="Map x2 cropped gas-liquid iSAT annotations back to source-image coordinates."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing x2 cropped iSAT JSON annotations.",
    )
    parser.add_argument(
        "--crop-params",
        type=Path,
        default=default_crop_params,
        help="crop_params.json generated for the original crop operation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for remapped source-space iSAT annotations.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale applied before adding crop offsets. Use 0.5 for x2 super-res outputs.",
    )
    parser.add_argument(
        "--source-folder",
        type=str,
        default=None,
        help="Optional override for info.folder in output JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    crop_params_path = args.crop_params.resolve()
    output_dir = args.output_dir.resolve()

    if args.scale <= 0:
        raise ValueError("--scale must be a positive number.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input annotation directory does not exist: {input_dir}")
    if not crop_params_path.is_file():
        raise FileNotFoundError(f"Crop params JSON does not exist: {crop_params_path}")

    crop_params = load_json(crop_params_path)
    crop_box = crop_params.get("crop_box", {})
    source_size = crop_params.get("source_size", {})

    offset_x = float(crop_box["x1"])
    offset_y = float(crop_box["y1"])
    source_width = int(source_size["width"])
    source_height = int(source_size["height"])

    source_folder = args.source_folder
    if source_folder is None:
        source_folder = crop_params.get("paths", {}).get("frame_dir")
        if source_folder is not None:
            source_folder = str(source_folder).replace("\\", "/")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    remapped_count = 0
    copied_json_count = 0

    for json_path in json_files:
        payload = load_json(json_path)
        if is_isat_annotation(payload):
            payload = remap_isat_annotation(
                payload=payload,
                scale=args.scale,
                offset_x=offset_x,
                offset_y=offset_y,
                source_width=source_width,
                source_height=source_height,
                source_folder=source_folder,
            )
            remapped_count += 1
        else:
            copied_json_count += 1
        write_json(output_dir / json_path.name, payload)

    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() != ".json":
            copy2(path, output_dir / path.name)

    print(f"Input annotations: {input_dir}")
    print(f"Crop params: {crop_params_path}")
    print(f"Output annotations: {output_dir}")
    print(f"Scale factor: {args.scale}")
    print(f"Offset: (x1={offset_x}, y1={offset_y})")
    print(f"Remapped iSAT JSON files: {remapped_count}")
    print(f"Copied non-iSAT JSON files: {copied_json_count}")


if __name__ == "__main__":
    main()