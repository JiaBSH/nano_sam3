#!/usr/bin/env python3
"""Split gas-liquid frames by mark intervals and keep only start-frame mark.

Workflow:
1. Split frame/mark files into group folders by mark index intervals.
2. In each group's mark folder, keep only the mark json of the start index.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

GROUP_PATTERN = re.compile(r"^group_\d+_(\d+)-(\d+)$")
MARK_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.json$")
FRAME_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split by mark intervals, then keep only start-frame mark"
    )
    parser.add_argument(
        "--frame-dir",
        default="data/TEM/gas-liquid/gas-liquid-frame-crop_x2",
        help="Source frame directory",
    )
    parser.add_argument(
        "--mark-dir",
        default="data/TEM/gas-liquid/gas-liquid-mark-crop_x2",
        help="Source mark directory",
    )
    parser.add_argument(
        "--split-root",
        default="data/TEM/gas-liquid/gas-liquid-split-by-mark_x2",
        help="Output root directory containing group_* folders",
    )
    parser.add_argument(
        "--no-reset-output",
        action="store_true",
        help="Do not clear existing output directory before splitting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show operations without copying/deleting files",
    )
    return parser.parse_args()


def extract_idx(file_path: Path, pattern: re.Pattern[str]) -> int | None:
    m = pattern.match(file_path.name)
    if not m:
        return None
    return int(m.group("idx"))


def split_by_mark_intervals(
    frame_dir: Path,
    mark_dir: Path,
    split_root: Path,
    dry_run: bool,
    reset_output: bool,
) -> int:
    frame_files = [p for p in frame_dir.iterdir() if p.is_file() and FRAME_PATTERN.match(p.name)]
    mark_files = [p for p in mark_dir.iterdir() if p.is_file() and MARK_PATTERN.match(p.name)]
    isat_yaml = mark_dir / "isat.yaml"

    if not frame_files:
        raise SystemExit(f"No frame files found in: {frame_dir}")
    if not mark_files:
        raise SystemExit(f"No mark files found in: {mark_dir}")

    if not isat_yaml.exists():
        print(f"Warning: isat.yaml not found, skip yaml copy: {isat_yaml}")

    frame_map: dict[int, Path] = {}
    for f in frame_files:
        idx = extract_idx(f, FRAME_PATTERN)
        if idx is None:
            continue
        frame_map[idx] = f

    mark_map: dict[int, Path] = {}
    mark_indices: list[int] = []
    for m in mark_files:
        idx = extract_idx(m, MARK_PATTERN)
        if idx is None:
            continue
        mark_map[idx] = m
        mark_indices.append(idx)

    if not mark_indices:
        raise SystemExit("No valid mark index found from mark files.")

    mark_indices = sorted(set(mark_indices))
    max_frame_idx = max(frame_map.keys())

    if split_root.exists() and reset_output:
        if dry_run:
            print(f"[DRY-RUN] remove output root: {split_root}")
        else:
            shutil.rmtree(split_root)

    if dry_run:
        print(f"[DRY-RUN] ensure output root: {split_root}")
    else:
        split_root.mkdir(parents=True, exist_ok=True)

    group_count = 0
    for i, start_idx in enumerate(mark_indices):
        if i < len(mark_indices) - 1:
            end_idx = mark_indices[i + 1] - 1
        else:
            end_idx = max_frame_idx
        if end_idx < start_idx:
            continue

        group_name = f"group_{i + 1:03d}_{start_idx:012d}-{end_idx:012d}"
        group_dir = split_root / group_name
        frame_out = group_dir / "frame"
        mark_out = group_dir / "mark"

        if dry_run:
            print(f"[DRY-RUN] create: {frame_out}")
            print(f"[DRY-RUN] create: {mark_out}")
        else:
            frame_out.mkdir(parents=True, exist_ok=True)
            mark_out.mkdir(parents=True, exist_ok=True)

        if isat_yaml.exists():
            yaml_dst = mark_out / isat_yaml.name
            if dry_run:
                print(f"[DRY-RUN] copy yaml: {isat_yaml} -> {yaml_dst}")
            else:
                shutil.copy2(isat_yaml, yaml_dst)

        for idx in range(start_idx, end_idx + 1):
            if idx in frame_map:
                src = frame_map[idx]
                dst = frame_out / src.name
                if dry_run:
                    print(f"[DRY-RUN] copy frame: {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)

            if idx in mark_map:
                src = mark_map[idx]
                dst = mark_out / src.name
                if dry_run:
                    print(f"[DRY-RUN] copy mark: {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)

        group_count += 1

    print(f"Split done. groups={group_count}")
    return group_count


def keep_only_start_mark(split_root: Path, dry_run: bool) -> tuple[int, int, int]:
    group_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])
    if not group_dirs:
        print("No group folders found.")
        return 0, 0, 0

    total_deleted = 0
    touched_groups = 0

    for group_dir in group_dirs:
        gm = GROUP_PATTERN.match(group_dir.name)
        if not gm:
            continue

        start_idx = int(gm.group(1))
        mark_dir = group_dir / "mark"
        if not mark_dir.exists() or not mark_dir.is_dir():
            continue

        mark_files = sorted([p for p in mark_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        if len(mark_files) <= 1:
            continue

        keep_file = None
        for f in mark_files:
            idx = extract_idx(f, MARK_PATTERN)
            if idx == start_idx:
                keep_file = f
                break

        if keep_file is None:
            indexed = [(extract_idx(f, MARK_PATTERN), f) for f in mark_files]
            indexed = [(idx, f) for idx, f in indexed if idx is not None]
            if not indexed:
                continue
            indexed.sort(key=lambda x: x[0])
            keep_file = indexed[0][1]

        deleted = 0
        for f in mark_files:
            if f == keep_file:
                continue
            if dry_run:
                print(f"[DRY-RUN] delete mark: {f}")
            else:
                f.unlink()
            deleted += 1

        if deleted > 0:
            touched_groups += 1
            total_deleted += deleted
            print(f"{group_dir.name}: keep {keep_file.name}, delete {deleted}")

    print(
        f"Keep-start done. processed_groups={len(group_dirs)}, touched_groups={touched_groups}, deleted_marks={total_deleted}"
    )
    return len(group_dirs), touched_groups, total_deleted


def main() -> int:
    args = parse_args()
    frame_dir = Path(args.frame_dir)
    mark_dir = Path(args.mark_dir)
    split_root = Path(args.split_root)

    if not frame_dir.exists() or not frame_dir.is_dir():
        raise SystemExit(f"frame dir does not exist or is not a directory: {frame_dir}")
    if not mark_dir.exists() or not mark_dir.is_dir():
        raise SystemExit(f"mark dir does not exist or is not a directory: {mark_dir}")

    split_by_mark_intervals(
        frame_dir=frame_dir,
        mark_dir=mark_dir,
        split_root=split_root,
        dry_run=args.dry_run,
        reset_output=not args.no_reset_output,
    )
    keep_only_start_mark(split_root=split_root, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
