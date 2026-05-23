#!/usr/bin/env python3
"""Plot COCO summary metrics by model.

This script creates three figures from assets/coco_summary.csv:
1) Combined COCO segmentation AP metrics (mAP, mAP50, mAP75)
2) Inference time
3) GPU memory usage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "figure.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot COCO summary charts by model name")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("assets/50x_unsup_summary.csv"),
        help="Path to coco_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/coco_eval_apmask/plots"),
        help="Directory to save output figures",
    )
    parser.add_argument("--split", type=str, default=None, help="Filter by split")
    parser.add_argument("--mode", type=str, default=None, help="Filter by mode")
    parser.add_argument(
        "--time-col",
        type=str,
        default="time",
        choices=["time", "mean_time_ms", "median_time_ms", "max_time_ms"],
        help="Column used for time plot",
    )
    parser.add_argument(
        "--memory-col",
        type=str,
        default="mean_peak_allocated_mb",
        choices=[
            "mean_peak_allocated_mb",
            "median_peak_allocated_mb",
            "max_peak_allocated_mb",
            "mean_peak_reserved_mb",
            "median_peak_reserved_mb",
            "max_peak_reserved_mb",
        ],
        help="Column used for memory plot",
    )
    return parser.parse_args()


def _rotate_xticks() -> None:
    plt.xticks(rotation=45, ha="right")


def _display_model_names(models: pd.Series) -> pd.Series:
    suffix = "_custom_coco_instance"
    return models.astype(str).str.replace(f"{suffix}$", "", regex=True)


def plot_map_metrics(df: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = list(range(len(df)))
    width = 0.26
    model_labels = _display_model_names(df["model"])

    ax.bar(
        [i - width for i in x],
        df["coco/segm_mAP"],
        width=width,
        label="segm_mAP",
    )
    ax.bar(
        x,
        df["coco/segm_mAP_50"],
        width=width,
        label="segm_mAP_50",
    )
    ax.bar(
        [i + width for i in x],
        df["coco/segm_mAP_75"],
        width=width,
        label="segm_mAP_75",
    )

    fig.suptitle("Segmentation AP Metrics by Model", y=0.97)
    ax.set_xlabel("Model")
    ax.set_ylabel("AP")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    _rotate_xticks()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_pixel_metrics(df: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = list(range(len(df)))
    width = 0.2
    model_labels = _display_model_names(df["model"])

    ax.bar(
        [i - 1.5 * width for i in x],
        df["pixel_mean_iou"],
        width=width,
        label="pixel_mean_iou",
    )
    ax.bar(
        [i - 0.5 * width for i in x],
        df["pixel_mean_precision"],
        width=width,
        label="pixel_mean_precision",
    )
    ax.bar(
        [i + 0.5 * width for i in x],
        df["pixel_mean_recall"],
        width=width,
        label="pixel_mean_recall",
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        df["pixel_mean_f1"],
        width=width,
        label="pixel_mean_f1",
    )

    fig.suptitle("Pixel Metrics by Model", y=0.97)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    _rotate_xticks()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_time(df: pd.DataFrame, out_file: Path, time_col: str) -> None:
    model_labels = _display_model_names(df["model"])
    values = df[time_col].astype(float).tolist()
    x = list(range(len(values)))

    sorted_vals = sorted(values)
    vmax = sorted_vals[-1]
    second_max = sorted_vals[-2] if len(sorted_vals) > 1 else sorted_vals[-1]
    use_broken_axis = len(values) > 1 and second_max > 0 and vmax > 2.0 * second_max

    if not use_broken_axis:
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(x, values, color="#4C78A8")
        ax.set_title(f"{time_col} by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel(time_col)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        _rotate_xticks()
        fig.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        return

    low_max = second_max * 1.15
    high_min = max(vmax * 0.85, low_max * 1.1)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(16, 8),
        gridspec_kw={"height_ratios": [1, 3]},
    )

    ax_top.bar(x, values, color="#4C78A8")
    ax_bottom.bar(x, values, color="#4C78A8")

    ax_bottom.set_ylim(0, low_max)
    ax_top.set_ylim(high_min, vmax * 1.05)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)

    d = 0.008
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_top.set_title(f"{time_col} by Model (broken y-axis)")
    ax_bottom.set_xlabel("Model")
    ax_bottom.set_ylabel(time_col)
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(model_labels)
    _rotate_xticks()

    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_memory(df: pd.DataFrame, out_file: Path, memory_col: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    model_labels = _display_model_names(df["model"])
    ax.bar(model_labels, df[memory_col], color="#F58518")

    ax.set_title(f"GPU Memory Usage ({memory_col}) by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Memory (MB)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _rotate_xticks()
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_plot_style()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    required = [
        "model",
        "coco/segm_mAP",
        "coco/segm_mAP_50",
        "coco/segm_mAP_75",
        "pixel_mean_iou",
        "pixel_mean_precision",
        "pixel_mean_recall",
        "pixel_mean_f1",
        args.time_col,
        args.memory_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.split is not None:
        if "split" not in df.columns:
            raise ValueError("Column 'split' not found in CSV")
        df = df[df["split"] == args.split]

    if args.mode is not None:
        if "mode" not in df.columns:
            raise ValueError("Column 'mode' not found in CSV")
        df = df[df["mode"] == args.mode]

    if df.empty:
        raise ValueError("No rows left after filtering. Check --split/--mode values.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_map_metrics(df, args.out_dir / "coco_map_metrics.png")
    plot_pixel_metrics(df, args.out_dir / "coco_pixel_metrics.png")
    plot_time(df, args.out_dir / f"coco_{args.time_col}.png", args.time_col)
    plot_memory(df, args.out_dir / f"coco_{args.memory_col}.png", args.memory_col)

    print("Saved:")
    print(args.out_dir / "coco_map_metrics.png")
    print(args.out_dir / "coco_pixel_metrics.png")
    print(args.out_dir / f"coco_{args.time_col}.png")
    print(args.out_dir / f"coco_{args.memory_col}.png")


if __name__ == "__main__":
    main()
