from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cus_tools.export_prediction_to_isat import save_masks_as_isat
from sam3.model_builder import build_sam3_video_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the first-frame iSAT annotation in each gas-liquid group to initialize "
            "SAM3 tracking and predict masks for all frames."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_cus/gas-liquid-split-by-mark_x2"),
        help="Dataset root containing group_*/frame and group_*/mark folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/gas-liquid-first-frame-sam3"),
        help="Directory used to store predicted annotations.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Optional subset of group directory names to process.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional SAM3 checkpoint path. Defaults to local cache if available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Only export instances whose sigmoid(object_score_logit) is >= threshold.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="Minimum polygon area kept in exported iSAT json.",
    )
    parser.add_argument(
        "--save-mask-png",
        action="store_true",
        help="Also save per-frame instance-id masks as PNG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders for processed groups.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Offload loaded frames to CPU memory.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Offload tracker state to CPU memory.",
    )
    return parser.parse_args()


def natural_frame_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def resolve_checkpoint_path(explicit_path: Path | None) -> str | None:
    if explicit_path is not None:
        return str(explicit_path)
    cached = Path("ms_cache/facebook/sam3/sam3.pt")
    if cached.is_file():
        return str(cached)
    raise RuntimeError(
        "Default checkpoint not found: ms_cache/facebook/sam3/sam3.pt. "
        "Please place the checkpoint there or pass --checkpoint explicitly."
    )


def discover_groups(data_root: Path, selected_groups: list[str] | None) -> list[Path]:
    group_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if selected_groups is None:
        return group_dirs
    selected = set(selected_groups)
    return [path for path in group_dirs if path.name in selected]


def list_frame_paths(frame_dir: Path) -> list[Path]:
    frame_paths = [
        path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    frame_paths.sort(key=natural_frame_key)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")
    return frame_paths


def list_annotation_paths(mark_dir: Path) -> list[Path]:
    json_paths = [
        path for path in mark_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"
    ]
    json_paths.sort(key=natural_frame_key)
    if not json_paths:
        raise RuntimeError(f"No annotation json found in {mark_dir}")
    return json_paths


def read_default_category_name(annotation_payload: dict[str, Any]) -> str:
    objects = annotation_payload.get("objects") or []
    for obj in objects:
        category_name = str(obj.get("category") or "").strip()
        if category_name and category_name != "__background__":
            return category_name
    return "object"


def build_instance_category_map(annotation_payload: dict[str, Any]) -> dict[int, str]:
    category_by_group: dict[int, str] = {}
    for obj in annotation_payload.get("objects") or []:
        group_id = obj.get("group")
        category_name = str(obj.get("category") or "").strip()
        if not isinstance(group_id, int) or group_id <= 0:
            continue
        if not category_name or category_name == "__background__":
            continue
        category_by_group[group_id] = category_name
    return category_by_group


def normalize_polygon(segmentation: Any) -> list[np.ndarray]:
    if not segmentation:
        return []

    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], (int, float)):
        points = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
        return [points] if len(points) >= 3 else []

    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        first = segmentation[0]
        if first and isinstance(first[0], (int, float)):
            points = np.asarray(segmentation, dtype=np.float32)
            return [points] if len(points) >= 3 else []

        polygons: list[np.ndarray] = []
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 3:
                polygons.append(points)
        return polygons

    return []


def annotation_to_instance_masks(annotation_payload: dict[str, Any]) -> OrderedDict[int, np.ndarray]:
    info = annotation_payload.get("info") or {}
    height = int(info["height"])
    width = int(info["width"])
    merged_masks: OrderedDict[int, np.ndarray] = OrderedDict()
    next_instance_id = 1

    for obj in annotation_payload.get("objects") or []:
        group_id = obj.get("group")
        if not isinstance(group_id, int) or group_id <= 0:
            while next_instance_id in merged_masks:
                next_instance_id += 1
            group_id = next_instance_id
            next_instance_id += 1

        mask = merged_masks.setdefault(group_id, np.zeros((height, width), dtype=np.uint8))
        polygons = normalize_polygon(obj.get("segmentation"))
        for polygon in polygons:
            polygon_i32 = np.round(polygon).astype(np.int32)
            cv2.fillPoly(mask, [polygon_i32], color=1)

    merged_masks = OrderedDict(
        (instance_id, mask.astype(bool)) for instance_id, mask in merged_masks.items() if mask.any()
    )
    if not merged_masks:
        raise RuntimeError("The first-frame annotation does not contain any valid polygons.")
    return merged_masks


def create_sequential_jpeg_frames(frame_paths: list[Path], temp_root: Path) -> Path:
    temp_frame_dir = temp_root / "jpeg_frames"
    temp_frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_path in enumerate(frame_paths):
        target_path = temp_frame_dir / f"{idx:05d}.jpg"
        image = Image.open(frame_path).convert("RGB")
        image.save(target_path, format="JPEG", quality=100, subsampling=0)
    return temp_frame_dir


def write_instance_mask_png(path: Path, instance_masks: OrderedDict[int, np.ndarray]) -> None:
    sample_mask = next(iter(instance_masks.values()))
    colored = np.zeros((*sample_mask.shape, 3), dtype=np.uint8)
    for instance_id, mask_bool in instance_masks.items():
        # Use a deterministic bright palette so the saved PNG is directly viewable.
        color = np.array(
            [
                (instance_id * 73) % 192 + 48,
                (instance_id * 151) % 192 + 48,
                (instance_id * 199) % 192 + 48,
            ],
            dtype=np.uint8,
        )
        colored[mask_bool] = color
    Image.fromarray(colored, mode="RGB").save(path)


def propagate_group(
    predictor: Any,
    group_dir: Path,
    output_root: Path,
    category_name_override: str | None,
    min_area: float,
    score_threshold: float,
    save_mask_png: bool,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> None:
    frame_dir = group_dir / "frame"
    mark_dir = group_dir / "mark"
    frame_paths = list_frame_paths(frame_dir)
    annotation_paths = list_annotation_paths(mark_dir)

    first_annotation = load_json(annotation_paths[0])
    category_name = category_name_override or read_default_category_name(first_annotation)
    instance_category_map = build_instance_category_map(first_annotation)
    first_frame_masks = annotation_to_instance_masks(first_annotation)

    output_group_dir = output_root / group_dir.name
    output_mark_dir = output_group_dir / "mark"
    output_mask_dir = output_group_dir / "mask_png"
    if output_group_dir.exists():
        shutil.rmtree(output_group_dir)
    output_mark_dir.mkdir(parents=True, exist_ok=True)
    if save_mask_png:
        output_mask_dir.mkdir(parents=True, exist_ok=True)

    isat_yaml_path = mark_dir / "isat.yaml"
    if isat_yaml_path.is_file():
        shutil.copy2(isat_yaml_path, output_mark_dir / "isat.yaml")

    # The first frame is the reference annotation, so export it directly instead of re-predicting it.
    first_frame_path = frame_paths[0]
    dump_json(output_mark_dir / f"{first_frame_path.stem}.json", first_annotation)
    if save_mask_png:
        write_instance_mask_png(output_mask_dir / f"{first_frame_path.stem}.png", first_frame_masks)

    with tempfile.TemporaryDirectory(prefix=f"sam3_{group_dir.name}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        jpeg_frame_dir = create_sequential_jpeg_frames(frame_paths, temp_dir)
        inference_state = predictor.init_state(
            video_path=str(jpeg_frame_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )

        for instance_id, mask in first_frame_masks.items():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(instance_id),
                mask=torch.from_numpy(mask),
            )

        if len(frame_paths) <= 1:
            return

        for frame_idx, obj_ids, _, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=1,
            max_frame_num_to_track=len(frame_paths) - 1,
            reverse=False,
            propagate_preflight=True,
        ):
            per_frame_masks: OrderedDict[int, np.ndarray] = OrderedDict()
            for obj_pos, obj_id in enumerate(obj_ids):
                mask_tensor = video_res_masks[obj_pos]
                mask_2d = (mask_tensor > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                per_frame_masks[int(obj_id)] = mask_2d

            if obj_scores is None:
                instance_scores = {int(obj_id): 1.0 for obj_id in obj_ids}
            else:
                probs = torch.sigmoid(obj_scores.detach().to(torch.float32)).cpu().numpy().reshape(-1)
                instance_scores = {
                    int(obj_id): float(probs[obj_pos]) for obj_pos, obj_id in enumerate(obj_ids)
                }

            frame_path = frame_paths[frame_idx]
            output_json_path = output_mark_dir / f"{frame_path.stem}.json"
            save_masks_as_isat(
                image_path=frame_path,
                instance_masks=list(per_frame_masks.values()),
                output_json_path=output_json_path,
                category_name=category_name,
                scores=[instance_scores[int(instance_id)] for instance_id in per_frame_masks],
                instance_ids=[int(instance_id) for instance_id in per_frame_masks],
                instance_categories=[
                    instance_category_map.get(int(instance_id), category_name)
                    for instance_id in per_frame_masks
                ],
                min_area=min_area,
                score_threshold=score_threshold,
            )

            if save_mask_png:
                write_instance_mask_png(output_mask_dir / f"{frame_path.stem}.png", per_frame_masks)


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    group_dirs = discover_groups(data_root, args.groups)
    if not group_dirs:
        raise RuntimeError(f"No group directories selected under {data_root}")

    if output_root.exists() and not args.overwrite:
        existing = [path for path in group_dirs if (output_root / path.name).exists()]
        if existing:
            existing_names = ", ".join(path.name for path in existing)
            raise RuntimeError(
                f"Output already exists for: {existing_names}. Re-run with --overwrite to replace them."
            )

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    sam3_model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    for group_dir in group_dirs:
        print(f"[SAM3] Processing {group_dir.name}")
        propagate_group(
            predictor=predictor,
            group_dir=group_dir,
            output_root=output_root,
            category_name_override=None,
            min_area=float(args.min_area),
            score_threshold=float(args.score_threshold),
            save_mask_png=bool(args.save_mask_png),
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            offload_state_to_cpu=bool(args.offload_state_to_cpu),
        )

    print(f"[SAM3] Finished. Outputs saved to {output_root}")


if __name__ == "__main__":
    main()