#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
EXPECTED_SHA256 = "54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target",
        default="data/yolo/coco8_ultralytics",
        help="Directory where the dataset should be placed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing dataset directory if it already exists.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official Ultralytics COCO8 smoke-test dataset.")
    configure_parser(parser)
    args = parser.parse_args()

    target_dir = Path(args.target).resolve()
    if target_dir.exists() and not args.force:
        raise SystemExit(f"Target already exists: {target_dir}. Use --force to replace it.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        archive_path = tmp_root / "coco8.zip"
        urllib.request.urlretrieve(DATASET_URL, archive_path)

        actual_sha = sha256sum(archive_path)
        if actual_sha != EXPECTED_SHA256:
            raise SystemExit(
                f"Unexpected SHA256 for downloaded archive: {actual_sha}. Expected: {EXPECTED_SHA256}."
            )

        extracted_root = tmp_root / "extracted"
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extracted_root)

        upstream_dir = extracted_root / "coco8"
        if not upstream_dir.exists():
            raise SystemExit("Downloaded archive does not contain the expected `coco8/` directory.")

        target_dir.mkdir(parents=True, exist_ok=True)

        for name in ("images", "labels", "LICENSE"):
            existing = target_dir / name
            if existing.is_dir():
                shutil.rmtree(existing)
            elif existing.exists():
                existing.unlink()
            source = upstream_dir / name
            destination = target_dir / name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        (target_dir / ".source.json").write_text(
            json.dumps(
                {
                    "dataset": "Ultralytics COCO8",
                    "download_url": DATASET_URL,
                    "zip_sha256": EXPECTED_SHA256,
                    "local_path": str(target_dir),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"Downloaded COCO8 into: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
