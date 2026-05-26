from pathlib import Path
import json
import shutil
import time

import pandas as pd
import torch
from PIL import Image

from sam3.agent.client_sam3 import sam3_inference
from sam3.agent.viz import visualize
from sam3.model.sam3_image_processor import Sam3Processor

from cus_tools.sam3_agent_eval_utils import expected_agent_paths
from cus_tools.export_prediction_to_isat import convert_prediction_json_to_isat


def save_prediction_artifacts(prediction_payload, prediction_path, visual_path):
    prediction_path = Path(prediction_path)
    visual_path = Path(visual_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    visual_path.parent.mkdir(parents=True, exist_ok=True)

    payload_to_save = {
        **prediction_payload,
        "output_image_path": str(visual_path.resolve()),
    }
    with prediction_path.open("w", encoding="utf-8") as handle:
        json.dump(payload_to_save, handle, indent=2, ensure_ascii=False)

    if prediction_payload["pred_masks"]:
        visualize(payload_to_save).save(visual_path)
    else:
        with Image.open(prediction_payload["original_image_path"]) as image_handle:
            image_handle.convert("RGB").save(visual_path)


def normalize_prediction_payload(prediction_payload):
    pred_scores = prediction_payload.get("pred_scores", [])
    pred_boxes = prediction_payload.get("pred_boxes", [])
    pred_masks = prediction_payload.get("pred_masks", [])

    score_indices = sorted(
        range(len(pred_scores)),
        key=lambda index: pred_scores[index],
        reverse=True,
    )
    pred_boxes = [pred_boxes[index] for index in score_indices]
    pred_masks = [pred_masks[index] for index in score_indices]
    pred_scores = [pred_scores[index] for index in score_indices]

    valid_indices = [index for index, rle in enumerate(pred_masks) if len(rle) > 4]
    prediction_payload["pred_boxes"] = [pred_boxes[index] for index in valid_indices]
    prediction_payload["pred_masks"] = [pred_masks[index] for index in valid_indices]
    prediction_payload["pred_scores"] = [pred_scores[index] for index in valid_indices]
    prediction_payload["original_image_path"] = str(
        Path(prediction_payload["original_image_path"]).resolve()
    )
    return prediction_payload


@torch.inference_mode()
def run_batch_text_prompt_inference(
    model,
    image_lookup,
    image_dir,
    text_prompt,
    model_name,
    output_dir,
    vis_dir,
    isat_output_dir=None,
    batch_size=4,
    confidence_threshold=0.5,
    category_name="畴区",
    min_area=20.0,
    polygon_simplify_epsilon=2.0,
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    vis_dir = Path(vis_dir)
    isat_output_dir = Path(isat_output_dir) if isat_output_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if isat_output_dir is not None:
        isat_output_dir.mkdir(parents=True, exist_ok=True)

    image_names = sorted(image_lookup)
    profile_rows = []
    processor = Sam3Processor(
        model,
        confidence_threshold=float(confidence_threshold) if confidence_threshold > 0 else 0.5,
    )

    for batch_start in range(0, len(image_names), batch_size):
        batch_names = image_names[batch_start : batch_start + batch_size]
        data_start = time.perf_counter()
        batch_image_paths = {
            image_name: (image_dir / image_name).resolve() for image_name in batch_names
        }
        for image_path in batch_image_paths.values():
            _ = image_path.stat()
        data_time_s = time.perf_counter() - data_start

        per_image_data_time_s = data_time_s / max(len(batch_names), 1)

        for image_name in batch_names:
            image_path = batch_image_paths[image_name]
            prediction_path, visual_path, _ = expected_agent_paths(
                image_path=image_path,
                text_prompt=text_prompt,
                llm_name=model_name,
                output_dir=output_dir,
            )

            num_predictions = 0
            status = "generated"
            error_message = ""
            isat_json_path = ""

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            infer_start = time.perf_counter()
            try:
                prediction_payload = sam3_inference(processor, str(image_path), text_prompt)
                prediction_payload["original_image_path"] = str(image_path)
                prediction_payload = normalize_prediction_payload(prediction_payload)
                save_prediction_artifacts(
                    prediction_payload=prediction_payload,
                    prediction_path=prediction_path,
                    visual_path=visual_path,
                )
                if isat_output_dir is not None:
                    isat_json_path = isat_output_dir / f"{image_path.stem}.json"
                    convert_prediction_json_to_isat(
                        input_json_path=prediction_path,
                        output_json_path=isat_json_path,
                        category_name=category_name,
                        score_threshold=confidence_threshold,
                        min_area=min_area,
                        polygon_simplify_epsilon=polygon_simplify_epsilon,
                    )
                num_predictions = len(prediction_payload["pred_masks"])
            except Exception as exc:
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            elapsed_s = time.perf_counter() - infer_start
            peak_allocated_mb = (
                float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
            )
            peak_reserved_mb = (
                float(torch.cuda.max_memory_reserved() / 1024**2) if torch.cuda.is_available() else 0.0
            )

            if visual_path.exists():
                shutil.copy2(visual_path, vis_dir / visual_path.name)

            profile_rows.append(
                {
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "prediction_json": str(prediction_path),
                    "prediction_isat": str(isat_json_path) if isat_json_path else "",
                    "prediction_visual": str(visual_path) if visual_path.exists() else "",
                    "num_predictions": num_predictions,
                    "time_s": elapsed_s,
                    "time_ms": elapsed_s * 1000.0,
                    "data_time_s": per_image_data_time_s,
                    "data_time_ms": per_image_data_time_s * 1000.0,
                    "peak_allocated_mb": peak_allocated_mb,
                    "peak_reserved_mb": peak_reserved_mb,
                    "status": status,
                    "attempt_count": 1,
                    "error_message": error_message,
                }
            )

    return pd.DataFrame(profile_rows)