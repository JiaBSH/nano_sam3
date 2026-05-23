from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether gas-liquid first-frame propagation outputs are complete for each group."
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
        help="Prediction root produced by predict_gas_liquid_from_first_frame.py.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Optional subset of group directory names to check.",
    )
    parser.add_argument(
        "--require-mask-png",
        action="store_true",
        help="Treat missing mask_png outputs as an error.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any mismatch is found.",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Also print groups that pass all checks.",
    )
    return parser.parse_args()


def discover_groups(data_root: Path, selected_groups: list[str] | None) -> list[Path]:
    group_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if selected_groups is None:
        return group_dirs
    selected = set(selected_groups)
    return [path for path in group_dirs if path.name in selected]


def stem_set_from_dir(path: Path, suffixes: set[str]) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.stem for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes}


def summarize_names(names: set[str], limit: int = 5) -> str:
    ordered = sorted(names)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    preview = ", ".join(ordered[:limit])
    return f"{preview}, ... (+{len(ordered) - limit} more)"


def check_group(group_dir: Path, output_root: Path, require_mask_png: bool) -> list[str]:
    issues: list[str] = []
    frame_dir = group_dir / "frame"
    output_group_dir = output_root / group_dir.name
    output_mark_dir = output_group_dir / "mark"
    output_mask_dir = output_group_dir / "mask_png"

    frame_stems = stem_set_from_dir(frame_dir, IMAGE_SUFFIXES)
    if not frame_stems:
        issues.append(f"no input frames under {frame_dir}")
        return issues

    if not output_group_dir.is_dir():
        issues.append(f"missing output group directory: {output_group_dir}")
        return issues

    if not output_mark_dir.is_dir():
        issues.append(f"missing mark directory: {output_mark_dir}")
    else:
        json_stems = stem_set_from_dir(output_mark_dir, {".json"})
        json_stems.discard("isat")
        missing_json = frame_stems - json_stems
        extra_json = json_stems - frame_stems
        if missing_json:
            issues.append(f"missing json for frames: {summarize_names(missing_json)}")
        if extra_json:
            issues.append(f"extra json without frames: {summarize_names(extra_json)}")
        if not (output_mark_dir / "isat.yaml").is_file():
            issues.append("missing copied isat.yaml")

    if require_mask_png:
        if not output_mask_dir.is_dir():
            issues.append(f"missing mask_png directory: {output_mask_dir}")
        else:
            png_stems = stem_set_from_dir(output_mask_dir, {".png"})
            missing_png = frame_stems - png_stems
            extra_png = png_stems - frame_stems
            if missing_png:
                issues.append(f"missing png for frames: {summarize_names(missing_png)}")
            if extra_png:
                issues.append(f"extra png without frames: {summarize_names(extra_png)}")

    return issues


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    group_dirs = discover_groups(data_root, args.groups)
    if not group_dirs:
        raise RuntimeError(f"No group directories selected under {data_root}")

    problem_count = 0
    for group_dir in group_dirs:
        issues = check_group(group_dir, output_root, require_mask_png=bool(args.require_mask_png))
        if issues:
            problem_count += 1
            print(f"[FAIL] {group_dir.name}")
            for issue in issues:
                print(f"  - {issue}")
        elif args.show_matches:
            print(f"[OK] {group_dir.name}")

    checked_count = len(group_dirs)
    passed_count = checked_count - problem_count
    print(
        f"Checked {checked_count} groups under {output_root}: "
        f"{passed_count} passed, {problem_count} failed."
    )

    if problem_count > 0 and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()