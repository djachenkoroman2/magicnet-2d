from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import re
import shutil
from typing import Sequence

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Source tiles are ~100MP orthophotos.

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MANAGED_OUTPUT_NAMES = ("images", "labels", "README.md", "metadata.json", "classes.txt")
CHIP_NAME_RE = re.compile(r"^(?P<stem>.+)__x(?P<index_x>\d+)_y(?P<index_y>\d+)$")
DEFAULT_MIN_CLIP_SIDE = 2.0
JPEG_QUALITY = 95


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class TileWindow:
    index_x: int
    index_y: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class SourceImage:
    split: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    boxes: tuple[YoloBox, ...]

    @property
    def stem(self) -> str:
        return self.image_path.stem


@dataclass(frozen=True)
class Chip:
    source_stem: str
    split: str
    window: TileWindow
    rows: tuple[str, ...]

    @property
    def chip_stem(self) -> str:
        return chip_name(self.source_stem, self.window.index_x, self.window.index_y)

    @property
    def is_empty(self) -> bool:
        return not self.rows


@dataclass
class TilingPlan:
    source_root: Path
    output_root: Path
    tile_size: int
    stride: int
    empty_frac: float
    seed: int
    sources: tuple[SourceImage, ...]
    chips: tuple[Chip, ...]
    classes: tuple[str, ...]
    source_box_counts: Counter[int]
    kept_box_counts: Counter[int]
    degenerate_dropped: int
    empty_chips_kept: int
    empty_chips_skipped: int
    estimated_bytes: int | None

    @property
    def chips_per_split(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for chip in self.chips:
            counts[chip.split] += 1
        return counts

    @property
    def total_source_boxes(self) -> int:
        return sum(self.source_box_counts.values())

    @property
    def total_kept_boxes(self) -> int:
        return sum(self.kept_box_counts.values())


def chip_name(source_stem: str, index_x: int, index_y: int) -> str:
    return f"{source_stem}__x{index_x}_y{index_y}"


def parse_chip_name(name: str) -> tuple[str, int, int]:
    match = CHIP_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid chip name: {name!r}. Expected '<source_stem>__x<index_x>_y<index_y>'.")
    return match.group("stem"), int(match.group("index_x")), int(match.group("index_y"))


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> tuple[YoloBox, ...]:
    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO annotation at {label_path}:{line_number}: {raw_line!r}.")
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO annotation at {label_path}:{line_number}: {raw_line!r}.") from exc
        class_value, x_center, y_center, width, height = values
        class_id = int(class_value)
        if class_value != class_id:
            raise ValueError(f"Class id must be an integer at {label_path}:{line_number}: {parts[0]!r}.")
        for field_name, field_value in (("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)):
            if not 0.0 <= field_value <= 1.0:
                raise ValueError(
                    f"{field_name} must be in [0, 1] at {label_path}:{line_number}: {field_value!r}."
                )
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Box size must be positive at {label_path}:{line_number}: {raw_line!r}.")
        x_min = (x_center - width / 2.0) * image_width
        y_min = (y_center - height / 2.0) * image_height
        x_max = (x_center + width / 2.0) * image_width
        y_max = (y_center + height / 2.0) * image_height
        boxes.append(
            YoloBox(
                class_id=class_id,
                x_min=max(0.0, x_min),
                y_min=max(0.0, y_min),
                x_max=min(float(image_width), x_max),
                y_max=min(float(image_height), y_max),
            )
        )
    return tuple(boxes)


def grid_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def grid_windows(image_width: int, image_height: int, tile_size: int, stride: int) -> tuple[TileWindow, ...]:
    windows: list[TileWindow] = []
    for index_y, y_min in enumerate(grid_starts(image_height, tile_size, stride)):
        for index_x, x_min in enumerate(grid_starts(image_width, tile_size, stride)):
            windows.append(
                TileWindow(
                    index_x=index_x,
                    index_y=index_y,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=min(image_width, x_min + tile_size),
                    y_max=min(image_height, y_min + tile_size),
                )
            )
    return tuple(windows)


def clip_boxes_to_window(
    boxes: Sequence[YoloBox],
    window: TileWindow,
    *,
    min_clip_side: float = DEFAULT_MIN_CLIP_SIDE,
) -> tuple[list[str], Counter[int], int]:
    rows: list[str] = []
    kept_counts: Counter[int] = Counter()
    degenerate = 0
    for box in boxes:
        x_min = max(box.x_min, float(window.x_min))
        y_min = max(box.y_min, float(window.y_min))
        x_max = min(box.x_max, float(window.x_max))
        y_max = min(box.y_max, float(window.y_max))
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0.0 or height <= 0.0:
            continue
        if width < min_clip_side or height < min_clip_side:
            degenerate += 1
            continue
        x_center = (x_min + width / 2.0 - window.x_min) / window.width
        y_center = (y_min + height / 2.0 - window.y_min) / window.height
        rows.append(f"{box.class_id} {x_center:.6f} {y_center:.6f} {width / window.width:.6f} {height / window.height:.6f}")
        kept_counts[box.class_id] += 1
    return rows, kept_counts, degenerate


def _discover_sources(source_root: Path) -> tuple[SourceImage, ...]:
    sources: list[SourceImage] = []
    for split in SPLITS:
        images_dir = source_root / "images" / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Split images directory does not exist: {images_dir}.")
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = source_root / "labels" / split / image_path.with_suffix(".txt").name
            if not label_path.is_file():
                raise FileNotFoundError(f"Label file is missing for image: {image_path} -> {label_path}.")
            with Image.open(image_path) as image:
                width, height = image.size
            sources.append(
                SourceImage(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    boxes=read_yolo_boxes(label_path, width, height),
                )
            )
    if not sources:
        raise ValueError(f"No source images found under: {source_root / 'images'}.")
    return tuple(sources)


def _select_empty_chip_count(total_windows: int, empty_count: int, empty_frac: float) -> int:
    if empty_count == 0:
        return 0
    target = int(round(empty_frac * total_windows))
    return min(empty_count, max(1, target))


def build_tiling_plan(
    *,
    source_root: Path,
    output_root: Path,
    tile_size: int = 1280,
    stride: int = 1024,
    empty_frac: float = 0.15,
    seed: int = 42,
    sample_images_for_estimate: int = 3,
) -> TilingPlan:
    if tile_size < 4:
        raise ValueError(f"tile_size must be at least 4 pixels, got {tile_size}.")
    if not 1 <= stride <= tile_size:
        raise ValueError(f"stride must be in [1, tile_size], got {stride} (tile_size={tile_size}).")
    if not 0.0 <= empty_frac <= 1.0:
        raise ValueError(f"empty_frac must be in [0, 1], got {empty_frac}.")
    if source_root.resolve() == output_root.resolve():
        raise ValueError(f"Source and output roots must differ: {source_root}.")

    classes_path = source_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"classes.txt does not exist under source root: {classes_path}.")
    classes = tuple(classes_path.read_text(encoding="utf-8").splitlines())

    sources = _discover_sources(source_root)
    chips: list[Chip] = []
    source_box_counts: Counter[int] = Counter()
    kept_box_counts: Counter[int] = Counter()
    degenerate_dropped = 0
    empty_chips_kept = 0
    empty_chips_skipped = 0

    for source in sources:
        for box in source.boxes:
            source_box_counts[box.class_id] += 1
        windows = grid_windows(source.width, source.height, tile_size, stride)
        prepared: list[tuple[TileWindow, tuple[str, ...], Counter[int], int]] = []
        empty_indices: list[int] = []
        for window_index, window in enumerate(windows):
            rows, counts, degenerate = clip_boxes_to_window(source.boxes, window)
            degenerate_dropped += degenerate
            kept_box_counts.update(counts)
            if rows:
                prepared.append((window, tuple(rows), counts, degenerate))
            else:
                empty_indices.append(window_index)
                prepared.append((window, (), counts, degenerate))
        keep_empty = _select_empty_chip_count(len(windows), len(empty_indices), empty_frac)
        rng = random.Random(f"{seed}:{source.split}:{source.stem}")
        kept_empty_indices = set(rng.sample(empty_indices, keep_empty)) if keep_empty else set()
        for window_index, (window, rows, _counts, _degenerate) in enumerate(prepared):
            if not rows and window_index not in kept_empty_indices:
                empty_chips_skipped += 1
                continue
            if not rows:
                empty_chips_kept += 1
            chips.append(Chip(source_stem=source.stem, split=source.split, window=window, rows=rows))

    estimated_bytes = _estimate_chip_bytes(sources, tile_size, stride, sample_images_for_estimate)

    return TilingPlan(
        source_root=source_root,
        output_root=output_root,
        tile_size=tile_size,
        stride=stride,
        empty_frac=empty_frac,
        seed=seed,
        sources=sources,
        chips=tuple(chips),
        classes=classes,
        source_box_counts=source_box_counts,
        kept_box_counts=kept_box_counts,
        degenerate_dropped=degenerate_dropped,
        empty_chips_kept=empty_chips_kept,
        empty_chips_skipped=empty_chips_skipped,
        estimated_bytes=estimated_bytes,
    )


def _estimate_chip_bytes(sources: Sequence[SourceImage], tile_size: int, stride: int, sample_count: int) -> int | None:
    sample = [source for source in sources if source.width >= tile_size // 2][:sample_count]
    if not sample:
        return None
    total_bytes = 0
    sampled_windows = 0
    for source in sample:
        with Image.open(source.image_path) as image:
            for window in grid_windows(source.width, source.height, tile_size, stride)[:8]:
                buffer = io.BytesIO()
                image.crop((window.x_min, window.y_min, window.x_max, window.y_max)).save(
                    buffer, format="JPEG", quality=JPEG_QUALITY
                )
                total_bytes += buffer.getbuffer().nbytes
                sampled_windows += 1
    if sampled_windows == 0:
        return None
    windows_total = sum(len(grid_windows(source.width, source.height, tile_size, stride)) for source in sources)
    return int(total_bytes / sampled_windows * windows_total)


def summarize_tiling(plan: TilingPlan) -> str:
    lines = [
        f"Source root: {plan.source_root}",
        f"Output root: {plan.output_root}",
        f"Tile size: {plan.tile_size}, stride: {plan.stride} (overlap {plan.tile_size - plan.stride} px, {100.0 * (plan.tile_size - plan.stride) / plan.tile_size:.0f}%)",
        f"Empty chip fraction: {plan.empty_frac} (seed {plan.seed})",
        f"Source images: {len(plan.sources)}",
        f"Chips to write: {len(plan.chips)}",
    ]
    for split in SPLITS:
        lines.append(f"  - {split}: {plan.chips_per_split.get(split, 0)}")
    lines.extend(
        [
            f"Source boxes: {plan.total_source_boxes}",
            f"Boxes kept on chips: {plan.total_kept_boxes}",
            f"Degenerate clipped boxes dropped: {plan.degenerate_dropped}",
            f"Empty chips kept: {plan.empty_chips_kept}, skipped: {plan.empty_chips_skipped}",
        ]
    )
    lines.append("Classes:")
    for class_id in sorted(set(plan.source_box_counts) | set(plan.kept_box_counts)):
        name = plan.classes[class_id] if class_id < len(plan.classes) else f"class {class_id}"
        lines.append(
            f"  - {class_id}: {name} (source {plan.source_box_counts.get(class_id, 0)}, tiles {plan.kept_box_counts.get(class_id, 0)})"
        )
    if plan.estimated_bytes is not None:
        lines.append(f"Estimated chip image size on disk: {plan.estimated_bytes / 1e6:.0f} MB (sample-based)")
    return "\n".join(lines)


def prepare_tiling(plan: TilingPlan, *, dry_run: bool, force: bool) -> None:
    if dry_run:
        return

    if plan.output_root.exists() and _has_existing_output(plan.output_root) and not force:
        raise FileExistsError(
            f"Output directory already contains files: {plan.output_root}. "
            "Re-run with --force to replace managed tiled dataset files."
        )
    if force:
        _remove_managed_output(plan.output_root)

    for split in SPLITS:
        for section in ("images", "labels"):
            (plan.output_root / section / split).mkdir(parents=True, exist_ok=True)

    chips_by_source: dict[tuple[str, str], list[Chip]] = {}
    for chip in plan.chips:
        chips_by_source.setdefault((chip.split, chip.source_stem), []).append(chip)

    written = 0
    for source in plan.sources:
        source_chips = chips_by_source.get((source.split, source.stem), [])
        if not source_chips:
            continue
        with Image.open(source.image_path) as image:
            for chip in source_chips:
                window = chip.window
                crop = image.crop((window.x_min, window.y_min, window.x_max, window.y_max))
                chip_image = plan.output_root / "images" / chip.split / f"{chip.chip_stem}{source.image_path.suffix.lower()}"
                crop.save(chip_image, format="JPEG", quality=JPEG_QUALITY)
                chip_label = plan.output_root / "labels" / chip.split / f"{chip.chip_stem}.txt"
                chip_label.write_text("\n".join(chip.rows) + ("\n" if chip.rows else ""), encoding="utf-8")
        written += 1
        if written % 100 == 0:
            print(f"  tiled {written}/{len(plan.sources)} source images...")

    shutil.copy2(plan.source_root / "classes.txt", plan.output_root / "classes.txt")
    (plan.output_root / "metadata.json").write_text(
        json.dumps(_metadata(plan), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (plan.output_root / "README.md").write_text(_render_readme(plan), encoding="utf-8")


def _has_existing_output(output_root: Path) -> bool:
    return any((output_root / name).exists() for name in MANAGED_OUTPUT_NAMES)


def _remove_managed_output(output_root: Path) -> None:
    for name in MANAGED_OUTPUT_NAMES:
        path = output_root / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _metadata(plan: TilingPlan) -> dict[str, object]:
    return {
        "source_root": str(plan.source_root),
        "output_root": str(plan.output_root),
        "created_by": "scripts/tile_energy_dataset.py",
        "tile_size": plan.tile_size,
        "stride": plan.stride,
        "overlap_px": plan.tile_size - plan.stride,
        "empty_frac": plan.empty_frac,
        "seed": plan.seed,
        "source_images": len(plan.sources),
        "splits": dict(plan.chips_per_split),
        "total_chips": len(plan.chips),
        "source_boxes": plan.total_source_boxes,
        "kept_boxes": plan.total_kept_boxes,
        "degenerate_clipped_boxes_dropped": plan.degenerate_dropped,
        "empty_chips_kept": plan.empty_chips_kept,
        "empty_chips_skipped": plan.empty_chips_skipped,
        "categories": [
            {
                "id": class_id,
                "name": plan.classes[class_id] if class_id < len(plan.classes) else f"class {class_id}",
                "annotations_source": plan.source_box_counts.get(class_id, 0),
                "annotations_tiles": plan.kept_box_counts.get(class_id, 0),
            }
            for class_id in sorted(set(plan.source_box_counts) | set(plan.kept_box_counts))
        ],
        "classes": list(plan.classes),
        "chip_naming": "<source_stem>__x<index_x>_y<index_y>",
        "geo_files": "not copied; restore georeferencing from the source image and the chip window offsets",
    }


def _render_readme(plan: TilingPlan) -> str:
    splits = plan.chips_per_split
    categories = _metadata(plan)["categories"]
    lines = [
        "# energy_obj_tiles",
        "",
        "Tiled variant of the prepared `energy_obj` dataset for small-object detector training.",
        "Produced by `scripts/tile_energy_dataset.py` from `data/energy_obj`; the source dataset is left unchanged.",
        "",
        "## Structure",
        "",
        "```text",
        "data/energy_obj_tiles/",
        "├── images/",
        "│   ├── train/        # JPEG chips",
        "│   ├── val/",
        "│   └── test/",
        "├── labels/",
        "│   ├── train/        # YOLO labels per chip (empty file = background chip)",
        "│   ├── val/",
        "│   └── test/",
        "├── classes.txt",
        "├── metadata.json",
        "└── README.md",
        "```",
        "",
        "Chips are named `<source_stem>__x<index_x>_y<index_y>.jpg`, so the source image and the",
        "pixel window can be recovered from the chip name alone.",
        "",
        "## Tiling parameters",
        "",
        f"- tile size: {plan.tile_size}×{plan.tile_size} px",
        f"- stride: {plan.stride} px (overlap {plan.tile_size - plan.stride} px)",
        f"- empty chip fraction: {plan.empty_frac} (seed {plan.seed})",
        "- edge windows are clamped to the source image bounds",
        f"- clipped boxes smaller than {DEFAULT_MIN_CLIP_SIDE:g} px on a side are dropped ({plan.degenerate_dropped} dropped)",
        "",
        "## Splits (chips)",
        "",
    ]
    for split in SPLITS:
        lines.append(f"- `{split}`: {splits.get(split, 0)} chips")
    lines.extend(
        [
            "",
            "Chips inherit the split of their source image, so train/val/test stay leak-free.",
            "",
            "## Classes",
            "",
        ]
    )
    for category in categories:
        lines.append(
            f"- `{category['id']}`: `{category['name']}` (source {category['annotations_source']} annotations, "
            f"tiles {category['annotations_tiles']})"
        )
    lines.extend(
        [
            "",
            "## Geo files",
            "",
            "`.jgw`/`.prj` files are **not** copied: training chips do not need them. To map predictions back to",
            "geo coordinates, decode the source stem and window from the chip name and combine with the source",
            "image world file from `data/energy_obj/geo/<split>/`.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "uv run python scripts/tile_energy_dataset.py --dry-run",
            "uv run python scripts/tile_energy_dataset.py --force",
            "```",
            "",
            "See `doc/training_recommendations.md` for the rationale behind the tiling parameters.",
        ]
    )
    return "\n".join(lines) + "\n"


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", default="data/energy_obj", help="Prepared flat YOLO dataset root.")
    parser.add_argument("--output-root", default="data/energy_obj_tiles", help="Tiled dataset output root.")
    parser.add_argument("--tile-size", type=int, default=1280, help="Chip edge length in pixels.")
    parser.add_argument("--stride", type=int, default=1024, help="Distance between adjacent chip origins in pixels.")
    parser.add_argument(
        "--empty-frac",
        type=float,
        default=0.15,
        help="Fraction of an image's chips kept as empty background chips.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic empty-chip selection seed.")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without writing files.")
    parser.add_argument("--force", action="store_true", help="Replace managed files under the output root.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tile-energy-dataset",
        description="Tile the prepared energy_obj YOLO dataset into a small-object training variant.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    plan = build_tiling_plan(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        tile_size=args.tile_size,
        stride=args.stride,
        empty_frac=args.empty_frac,
        seed=args.seed,
    )
    print(summarize_tiling(plan))
    if args.dry_run:
        print("Dry run only; no files were written.")
        return 0
    prepare_tiling(plan, dry_run=False, force=args.force)
    print(f"Tiled dataset written to: {plan.output_root}")
    return 0
