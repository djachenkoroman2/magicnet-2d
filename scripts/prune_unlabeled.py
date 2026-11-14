#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.energy_dataset import prune_empty_label_items


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="prune-unlabeled",
        description="Move images without annotations out of the prepared energy dataset into a holding root for re-annotation.",
    )
    parser.add_argument("--dataset-root", default="data/energy_obj", help="Prepared dataset root.")
    parser.add_argument(
        "--output-root",
        default="data/energy_obj_unlabeled",
        help="Holding root for images without annotations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without moving files.")
    args = parser.parse_args()

    summary = prune_empty_label_items(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        dry_run=args.dry_run,
    )
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Unlabeled images: {summary['pruned_images']}")
    for split_name, count in summary["pruned_by_split"].items():
        print(f"  - {split_name}: {count}")
    if args.dry_run:
        print("Dry run only; no files were moved.")
        return 0
    if summary["pruned_images"] == 0:
        print("Nothing to prune.")
        return 0
    print(f"Unlabeled images moved to: {summary['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
