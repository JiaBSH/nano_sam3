from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")


def sync_pin_annotations(source_dir: Path, target_dir: Path) -> dict[str, int]:
    stats = {
        "updated": 0,
        "missing_source": 0,
        "missing_target": 0,
        "source_without_pin": 0,
    }

    source_files = {path.name: path for path in source_dir.glob("*.json")}
    target_files = {path.name: path for path in target_dir.glob("*.json")}

    for name, target_path in sorted(target_files.items()):
        source_path = source_files.get(name)
        if source_path is None:
            stats["missing_source"] += 1
            continue

        source_payload = load_json(source_path)
        target_payload = load_json(target_path)

        source_objects = source_payload.get("objects", [])
        target_objects = target_payload.get("objects", [])

        pin_objects = [copy.deepcopy(obj) for obj in source_objects if obj.get("category") == "pin"]
        if not pin_objects:
            stats["source_without_pin"] += 1
            continue

        filtered_target_objects = [obj for obj in target_objects if obj.get("category") != "pin"]
        target_payload["objects"] = filtered_target_objects + pin_objects
        write_json(target_path, target_payload)
        stats["updated"] += 1

    missing_target = sorted(set(source_files) - set(target_files))
    stats["missing_target"] = len(missing_target)
    return stats


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    default_source = root_dir / "data" / "defect_label"
    default_target = root_dir / "data" / "20260508-mark"

    parser = argparse.ArgumentParser(
        description="Copy pin annotations from defect_label JSON files into matching 20260508-mark JSON files."
    )
    parser.add_argument("--source", type=Path, default=default_source, help="Directory containing source JSON files.")
    parser.add_argument("--target", type=Path, default=default_target, help="Directory containing target JSON files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    target_dir = args.target.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    stats = sync_pin_annotations(source_dir, target_dir)
    print(f"Updated files: {stats['updated']}")
    print(f"Missing source JSON: {stats['missing_source']}")
    print(f"Missing target JSON: {stats['missing_target']}")
    print(f"Source JSON without pin: {stats['source_without_pin']}")


if __name__ == "__main__":
    main()