from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
        f.write("\n")


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    p = np.asarray(points, dtype=np.float32)
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_bbox(points: list[list[float]]) -> list[float]:
    p = np.asarray(points, dtype=np.float32)
    x_min = float(np.min(p[:, 0]))
    y_min = float(np.min(p[:, 1]))
    x_max = float(np.max(p[:, 0]))
    y_max = float(np.max(p[:, 1]))
    return [x_min, y_min, x_max, y_max]


def _mask_to_polygons(mask_bool: np.ndarray) -> list[list[list[float]]]:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[list[list[float]]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour.reshape(-1, 2).astype(float).tolist()
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def _decode_rle_to_mask(rle_counts: str, height: int, width: int) -> np.ndarray:
    rle = {"counts": rle_counts, "size": [height, width]}
    decoded = mask_util.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool)


def _build_isat_payload(
    image_path: str | Path,
    masks: np.ndarray,
    scores: list[float],
    category_name: str,
    score_threshold: float,
    min_area: float,
    instance_ids: list[int] | None = None,
    instance_categories: list[str] | None = None,
) -> dict[str, Any]:
    img_path = Path(image_path)
    h = int(masks.shape[1])
    w = int(masks.shape[2])

    objects: list[dict[str, Any]] = []
    layer_idx = 1
    if instance_ids is None:
        resolved_instance_ids = [mask_idx + 1 for mask_idx in range(int(masks.shape[0]))]
    else:
        resolved_instance_ids = [int(instance_id) for instance_id in instance_ids]
    if instance_categories is None:
        resolved_instance_categories = [category_name for _ in range(int(masks.shape[0]))]
    else:
        resolved_instance_categories = [str(instance_category) for instance_category in instance_categories]

    for mask_idx, (mask_bool, score, instance_id) in enumerate(
        zip(masks, scores, resolved_instance_ids)
    ):
        if float(score) < score_threshold:
            continue

        instance_category = resolved_instance_categories[mask_idx]

        polygons = _mask_to_polygons(mask_bool)
        for poly in polygons:
            area = _polygon_area(poly)
            if area < min_area:
                continue
            objects.append(
                {
                    "category": instance_category,
                    "group": int(instance_id),
                    "segmentation": poly,
                    "area": area,
                    "layer": float(layer_idx),
                    "bbox": _polygon_bbox(poly),
                    "iscrowd": False,
                    "note": f"score={float(score):.4f},mask_id={mask_idx},instance_id={instance_id}",
                }
            )
            layer_idx += 1

    return {
        "info": {
            "description": "ISAT",
            "folder": str(img_path.parent.resolve()).replace("\\", "/"),
            "name": img_path.name,
            "width": w,
            "height": h,
            "depth": 3,
            "note": "",
        },
        "objects": objects,
    }


def save_inference_as_isat(
    image_path: str | Path,
    inference_state: dict[str, Any],
    output_json_path: str | Path,
    category_name: str = "畴区",
    score_threshold: float = 0.5,
    min_area: float = 20.0,
) -> Path:
    masks = (
        inference_state["masks"].squeeze(1).detach().cpu().numpy().astype(bool)
    )
    scores = (
        inference_state["scores"].detach().to(torch.float32).cpu().numpy().tolist()
    )

    payload = _build_isat_payload(
        image_path=image_path,
        masks=masks,
        scores=scores,
        category_name=category_name,
        score_threshold=float(score_threshold),
        min_area=float(min_area),
        instance_ids=None,
        instance_categories=None,
    )

    output_path = Path(output_json_path)
    _write_json(output_path, payload)
    return output_path


def save_masks_as_isat(
    image_path: str | Path,
    instance_masks: list[np.ndarray] | np.ndarray,
    output_json_path: str | Path,
    category_name: str = "畴区",
    scores: list[float] | None = None,
    score_threshold: float = 0.5,
    min_area: float = 20.0,
    instance_ids: list[int] | None = None,
    instance_categories: list[str] | None = None,
) -> Path:
    masks = np.asarray(instance_masks, dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]

    if scores is None:
        score_list = [1.0] * int(masks.shape[0])
    else:
        score_list = [float(score) for score in scores]

    payload = _build_isat_payload(
        image_path=image_path,
        masks=masks,
        scores=score_list,
        category_name=category_name,
        score_threshold=float(score_threshold),
        min_area=float(min_area),
        instance_ids=instance_ids,
        instance_categories=instance_categories,
    )

    output_path = Path(output_json_path)
    _write_json(output_path, payload)
    return output_path


def convert_prediction_json_to_isat(
    input_json_path: str | Path,
    output_json_path: str | Path | None = None,
    category_name: str = "畴区",
    score_threshold: float = 0.5,
    min_area: float = 20.0,
) -> Path:
    input_path = Path(input_json_path)
    data = _load_json(input_path)

    image_path = data.get("original_image_path") or data.get("image_path")
    if not image_path:
        raise ValueError("Input JSON must contain 'original_image_path' or 'image_path'.")

    h = int(data["orig_img_h"])
    w = int(data["orig_img_w"])
    pred_masks = data.get("pred_masks", [])
    pred_scores = data.get("pred_scores", [])

    if len(pred_masks) != len(pred_scores):
        raise ValueError(
            f"Length mismatch: pred_masks={len(pred_masks)} vs pred_scores={len(pred_scores)}"
        )

    if pred_masks:
        decoded_masks = np.stack(
            [_decode_rle_to_mask(rle, h, w) for rle in pred_masks], axis=0
        )
    else:
        decoded_masks = np.zeros((0, h, w), dtype=bool)

    payload = _build_isat_payload(
        image_path=image_path,
        masks=decoded_masks,
        scores=[float(s) for s in pred_scores],
        category_name=category_name,
        score_threshold=float(score_threshold),
        min_area=float(min_area),
        instance_ids=None,
        instance_categories=None,
    )

    if output_json_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_isat.json")
    else:
        output_path = Path(output_json_path)

    _write_json(output_path, payload)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAM3 prediction JSON to iSAT JSON annotation."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to prediction JSON with fields like pred_masks/pred_scores/orig_img_h/orig_img_w.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to output iSAT JSON. Default: <input_stem>_isat.json in same folder.",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="畴区",
        help="Category name written into iSAT objects.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Only keep masks with score >= this value.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Only keep polygons with area >= this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = convert_prediction_json_to_isat(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        category_name=args.category_name,
        score_threshold=args.score_threshold,
        min_area=args.min_area,
    )
    print(f"Saved iSAT annotation: {output_path}")


if __name__ == "__main__":
    main()
