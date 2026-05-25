from pathlib import Path
import json

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


def evaluate_coco_predictions(gt_path, predictions_path):
    predictions_path = Path(predictions_path)
    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    if not predictions:
        zero_metrics = {
            f"coco_eval_segm_{metric_name}": 0.0
            for metric_name in (
                "AP",
                "AP_50",
                "AP_75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "AR_maxDets@1",
                "AR_maxDets@10",
                "AR_maxDets@100",
                "AR_small",
                "AR_medium",
                "AR_large",
            )
        }
        return zero_metrics

    evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
        gt_path=str(gt_path),
        tide=False,
        iou_type="segm",
    )
    return evaluator.evaluate(str(predictions_path))