from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import pycocotools.mask as mask_utils

from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators


def flatten_isat_segmentation(segmentation):
    if not segmentation:
        return []

    first_item = segmentation[0]
    if first_item and isinstance(first_item[0], (int, float)):
        polygons = [segmentation]
    else:
        polygons = segmentation

    return [[coord for point in polygon for coord in point] for polygon in polygons if polygon]


def polygon_bbox(polygons):
    xs = []
    ys = []
    for polygon in polygons:
        xs.extend(polygon[0::2])
        ys.extend(polygon[1::2])

    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    return [float(xmin), float(ymin), float(max(0.0, xmax - xmin)), float(max(0.0, ymax - ymin))]


def build_coco_ground_truth(label_dir, output_path):
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "畴区", "supercategory": "畴区"}],
    }
    image_lookup = {}
    annotation_id = 1

    for image_id, label_path in enumerate(sorted(Path(label_dir).glob("*.json")), start=1):
        with label_path.open("r", encoding="utf-8") as handle:
            label_data = json.load(handle)

        info = label_data["info"]
        image_name = info["name"]
        width = int(info["width"])
        height = int(info["height"])

        dataset["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )
        image_lookup[image_name] = {
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
        }

        for obj in label_data.get("objects", []):
            polygons = flatten_isat_segmentation(obj.get("segmentation", []))
            if not polygons:
                continue

            bbox_raw = obj.get("bbox")
            if (
                bbox_raw
                and len(bbox_raw) == 4
                and bbox_raw[2] > bbox_raw[0]
                and bbox_raw[3] > bbox_raw[1]
            ):
                bbox = [
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2] - bbox_raw[0]),
                    float(bbox_raw[3] - bbox_raw[1]),
                ]
            else:
                bbox = polygon_bbox(polygons)

            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": float(obj.get("area", bbox[2] * bbox[3])),
                    "bbox": bbox,
                    "iscrowd": int(bool(obj.get("iscrowd", False))),
                }
            )
            annotation_id += 1

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)

    return dataset, image_lookup


def sanitize_prompt_for_filename(text_prompt):
    return text_prompt.replace("/", "_").replace(" ", "_")


def expected_agent_paths(image_path, text_prompt, llm_name, output_dir):
    image_basename = Path(image_path).stem
    prompt_stub = sanitize_prompt_for_filename(text_prompt)
    base_filename = f"{image_basename}_{prompt_stub}_agent_{llm_name}"
    output_dir = Path(output_dir)
    return (
        output_dir / f"{base_filename}_pred.json",
        output_dir / f"{base_filename}_pred.png",
        output_dir / f"{base_filename}_history.json",
    )


def normalized_cxcywh_to_xywh(box, image_width, image_height):
    center_x, center_y, width, height = box
    x1 = max(0.0, (center_x - width / 2.0) * image_width)
    y1 = max(0.0, (center_y - height / 2.0) * image_height)
    x2 = min(float(image_width), (center_x + width / 2.0) * image_width)
    y2 = min(float(image_height), (center_y + height / 2.0) * image_height)
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def load_agent_prediction_as_coco(prediction_path, image_info, next_annotation_id):
    with Path(prediction_path).open("r", encoding="utf-8") as handle:
        prediction = json.load(handle)

    annotations = []
    image_height = image_info["height"]
    image_width = image_info["width"]
    pred_scores = prediction.get("pred_scores", [])
    pred_masks = prediction.get("pred_masks", [])
    pred_boxes = prediction.get("pred_boxes", [])

    for index, counts in enumerate(pred_masks):
        if not counts:
            continue

        rle_for_tools = {
            "size": [image_height, image_width],
            "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
        }
        area = float(mask_utils.area(rle_for_tools))
        if area <= 0:
            continue

        if index < len(pred_boxes):
            bbox = normalized_cxcywh_to_xywh(pred_boxes[index], image_width, image_height)
        else:
            bbox = [float(value) for value in mask_utils.toBbox(rle_for_tools).tolist()]

        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_info["id"],
                "category_id": 1,
                "segmentation": {
                    "size": [image_height, image_width],
                    "counts": counts if isinstance(counts, str) else counts.decode("utf-8"),
                },
                "score": float(pred_scores[index]) if index < len(pred_scores) else 1.0,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
        )
        next_annotation_id += 1

    return annotations, next_annotation_id


def annotation_to_binary_mask(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return None

    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if counts is None:
            return None
        rle = {
            "size": segmentation.get("size", [height, width]),
            "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
        }
    elif isinstance(segmentation, list):
        if not segmentation:
            return None
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    else:
        return None

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def compute_pixel_mask_metrics(gt_dataset, predictions, pixel_summary_csv_path):
    import numpy as np

    images_by_id = {image_info["id"]: image_info for image_info in gt_dataset["images"]}
    gt_by_image = {}
    pred_by_image = {}

    for annotation in gt_dataset.get("annotations", []):
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation)

    for annotation in predictions:
        pred_by_image.setdefault(annotation["image_id"], []).append(annotation)

    pixel_rows = []

    for image_id, image_info in images_by_id.items():
        height = int(image_info["height"])
        width = int(image_info["width"])
        gt_union = np.zeros((height, width), dtype=bool)
        pred_union = np.zeros((height, width), dtype=bool)

        for annotation in gt_by_image.get(image_id, []):
            mask = annotation_to_binary_mask(annotation, height, width)
            if mask is not None:
                gt_union |= mask

        for annotation in pred_by_image.get(image_id, []):
            mask = annotation_to_binary_mask(annotation, height, width)
            if mask is not None:
                pred_union |= mask

        true_positive = int((gt_union & pred_union).sum())
        false_positive = int((~gt_union & pred_union).sum())
        false_negative = int((gt_union & ~pred_union).sum())

        denom_iou = true_positive + false_positive + false_negative
        denom_precision = true_positive + false_positive
        denom_recall = true_positive + false_negative
        iou = float(true_positive / denom_iou) if denom_iou > 0 else 0.0
        precision = float(true_positive / denom_precision) if denom_precision > 0 else 0.0
        recall = float(true_positive / denom_recall) if denom_recall > 0 else 0.0
        f1_score = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        pixel_rows.append(
            {
                "image_id": int(image_id),
                "image_name": image_info.get("file_name", str(image_id)),
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "pred_coverage": float(pred_union.mean()),
                "gt_coverage": float(gt_union.mean()),
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
            }
        )

    pixel_df = pd.DataFrame(pixel_rows)
    pixel_df.to_csv(pixel_summary_csv_path, index=False)

    if pixel_df.empty:
        summary = {
            "pixel_mean_iou": 0.0,
            "pixel_mean_precision": 0.0,
            "pixel_mean_recall": 0.0,
            "pixel_mean_f1": 0.0,
            "pixel_mean_pred_coverage": 0.0,
            "pixel_mean_gt_coverage": 0.0,
            "pixel_num_images": 0,
        }
    else:
        summary = {
            "pixel_mean_iou": float(pixel_df["iou"].mean()),
            "pixel_mean_precision": float(pixel_df["precision"].mean()),
            "pixel_mean_recall": float(pixel_df["recall"].mean()),
            "pixel_mean_f1": float(pixel_df["f1"].mean()),
            "pixel_mean_pred_coverage": float(pixel_df["pred_coverage"].mean()),
            "pixel_mean_gt_coverage": float(pixel_df["gt_coverage"].mean()),
            "pixel_num_images": int(len(pixel_df)),
        }

    return summary, pixel_df


def save_profile_artifacts(
    profile_df,
    profile_csv_path,
    profile_summary_json_path,
    profile_time_plot,
    profile_memory_plot,
):
    profile_df.to_csv(profile_csv_path, index=False)

    summary = {
        "mean_time_ms": float(profile_df["time_ms"].mean()),
        "median_time_ms": float(profile_df["time_ms"].median()),
        "max_time_ms": float(profile_df["time_ms"].max()),
        "mean_peak_allocated_mb": float(profile_df["peak_allocated_mb"].mean()),
        "median_peak_allocated_mb": float(profile_df["peak_allocated_mb"].median()),
        "max_peak_allocated_mb": float(profile_df["peak_allocated_mb"].max()),
        "mean_peak_reserved_mb": float(profile_df["peak_reserved_mb"].mean()),
        "median_peak_reserved_mb": float(profile_df["peak_reserved_mb"].median()),
        "max_peak_reserved_mb": float(profile_df["peak_reserved_mb"].max()),
        "num_images": int(len(profile_df)),
        "num_warmup": 0,
        "num_failed_images": int((profile_df["status"] == "failed").sum()),
    }

    with Path(profile_summary_json_path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    plt.figure(figsize=(max(6, len(profile_df) * 1.4), 4))
    plt.bar(profile_df["image_name"], profile_df["time_ms"], color="#2d6a4f")
    plt.ylabel("Time (ms)")
    plt.title("SAM3 Agent per-image latency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(profile_time_plot, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(max(6, len(profile_df) * 1.4), 4))
    bar_positions = range(len(profile_df))
    plt.bar(
        [position - 0.18 for position in bar_positions],
        profile_df["peak_allocated_mb"],
        width=0.36,
        label="allocated",
        color="#40916c",
    )
    plt.bar(
        [position + 0.18 for position in bar_positions],
        profile_df["peak_reserved_mb"],
        width=0.36,
        label="reserved",
        color="#74c69d",
    )
    plt.xticks(list(bar_positions), profile_df["image_name"], rotation=45, ha="right")
    plt.ylabel("Memory (MB)")
    plt.title("SAM3 Agent peak GPU memory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(profile_memory_plot, dpi=200, bbox_inches="tight")
    plt.close()

    return summary


def evaluate_coco_predictions(gt_path, predictions_path):
    evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
        gt_path=str(gt_path),
        tide=False,
        iou_type="segm",
    )
    return evaluator.evaluate(str(predictions_path))
