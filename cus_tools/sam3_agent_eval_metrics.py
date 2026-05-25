from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import pycocotools.mask as mask_utils


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