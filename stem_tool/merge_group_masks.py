from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SOURCE_TARGET_PAIRS = (
    ("frame", "frame", {".png"}),
    ("mask", "mask", {".json"}),
    ("mark", "mask", {".json"}),
    ("mask_png", "mask_png", {".png"}),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_source = project_root / "outputs" / "gas-liquid-first-frame-sam3"

    parser = argparse.ArgumentParser(
        description=(
            "Merge frame, mark/mask JSON, and mask_png files from each group_* directory "
            "into top-level folders."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="Root directory containing group_* subdirectories.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Whether to copy files into the merged folders or move them there.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the merged folders instead of failing on name collisions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without copying or moving files.",
    )
    return parser.parse_args()


def transfer_file(source_path: Path, target_path: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    if mode == "copy":
        shutil.copy2(source_path, target_path)
        return

    shutil.move(str(source_path), str(target_path))


def merge_category(
    source_root: Path,
    source_name: str,
    target_name: str,
    allowed_suffixes: set[str],
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, int]:
    stats = {
        "groups_with_source": 0,
        "files_merged": 0,
        "files_overwritten": 0,
    }

    pending_files: list[tuple[Path, Path]] = []

    for group_dir in sorted(path for path in source_root.glob("group_*") if path.is_dir()):
        category_dir = group_dir / source_name
        if not category_dir.is_dir():
            continue

        stats["groups_with_source"] += 1
        for source_path in sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        ):
            pending_files.append((source_path, source_root / target_name / source_path.name))

    if not pending_files:
        return stats

    target_dir = source_root / target_name
    if not dry_run:
        target_dir.mkdir(exist_ok=True)

    for source_path, target_path in pending_files:
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Target file already exists, use --overwrite to replace it: {target_path}"
            )

        if target_path.exists() and overwrite:
            stats["files_overwritten"] += 1

        transfer_file(source_path, target_path, mode=mode, dry_run=dry_run)
        stats["files_merged"] += 1

    return stats


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    for source_name, target_name, allowed_suffixes in SOURCE_TARGET_PAIRS:
        stats = merge_category(
            source_root,
            source_name,
            target_name,
            allowed_suffixes,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if stats["groups_with_source"] == 0:
            print(f"{source_name} -> {target_name}: skipped (no source directories found)")
            continue

        print(
            f"{source_name} -> {target_name}: groups={stats['groups_with_source']}, "
            f"merged={stats['files_merged']}, overwritten={stats['files_overwritten']}"
        )


if __name__ == "__main__":
    main()