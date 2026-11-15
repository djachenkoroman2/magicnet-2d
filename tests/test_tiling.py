from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from magicnet_2d.tiling import (
    YoloBox,
    _select_empty_chip_count,
    build_tiling_plan,
    chip_name,
    clip_boxes_to_window,
    grid_windows,
    main,
    parse_chip_name,
    read_yolo_boxes,
)


def _write_source_dataset(root: Path, size: tuple[int, int] = (20, 20)) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("substation\nsupport\n", encoding="utf-8")
    boxes_by_split = {
        "train": ["1 0.5 0.5 0.4 0.4\n", "0 0.1 0.1 0.1 0.1\n"],
        "val": ["1 0.75 0.25 0.2 0.2\n"],
        "test": ["1 0.5 0.5 0.3 0.3\n"],
    }
    for split, rows in boxes_by_split.items():
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        Image.new("RGB", size, color=(30, 60, 90)).save(images_dir / f"{split}-0.jpg", format="JPEG")
        (labels_dir / f"{split}-0.txt").write_text("".join(rows), encoding="utf-8")


def _snapshot_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums[path.relative_to(root).as_posix()] = hashlib.md5(path.read_bytes()).hexdigest()
    return checksums


class ChipNameTests(unittest.TestCase):
    def test_chip_name_roundtrip(self) -> None:
        name = chip_name("Alikino-15-25", 3, 7)
        self.assertEqual(name, "Alikino-15-25__x3_y7")
        self.assertEqual(parse_chip_name(name), ("Alikino-15-25", 3, 7))

    def test_parse_chip_name_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            parse_chip_name("Alikino-15-25")


class GridWindowTests(unittest.TestCase):
    def test_grid_covers_edges_with_clamped_last_window(self) -> None:
        windows = grid_windows(20, 20, tile_size=8, stride=6)
        self.assertEqual(len(windows), 9)
        xs = {window.x_min for window in windows}
        ys = {window.y_min for window in windows}
        self.assertEqual(xs, {0, 6, 12})
        self.assertEqual(ys, {0, 6, 12})
        edge_windows = [window for window in windows if window.x_min == 12]
        self.assertTrue(all(window.x_max == 20 for window in edge_windows))
        self.assertTrue(all(window.width == 8 for window in windows))

    def test_single_window_when_image_fits_tile(self) -> None:
        windows = grid_windows(8, 6, tile_size=8, stride=8)
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].x_min, windows[0].y_min, windows[0].x_max, windows[0].y_max), (0, 0, 8, 6))


class ClipBoxTests(unittest.TestCase):
    def test_center_box_stays_put(self) -> None:
        box = YoloBox(class_id=1, x_min=2.0, y_min=2.0, x_max=6.0, y_max=6.0)
        window = grid_windows(20, 20, 8, 8)[0]
        rows, counts, degenerate = clip_boxes_to_window([box], window)
        self.assertEqual(len(rows), 1)
        parts = [float(value) for value in rows[0].split()]
        self.assertAlmostEqual(parts[1], 0.5)
        self.assertAlmostEqual(parts[2], 0.5)
        self.assertAlmostEqual(parts[3], 0.5)
        self.assertAlmostEqual(parts[4], 0.5)
        self.assertEqual(counts[1], 1)
        self.assertEqual(degenerate, 0)

    def test_border_box_is_clipped_into_each_adjacent_chip(self) -> None:
        box = YoloBox(class_id=2, x_min=6.0, y_min=6.0, x_max=10.0, y_max=10.0)
        windows = grid_windows(16, 16, 8, 8)
        rows_by_corner = {}
        for window in windows:
            rows, _counts, _degenerate = clip_boxes_to_window([box], window)
            if rows:
                rows_by_corner[(window.index_x, window.index_y)] = [float(v) for v in rows[0].split()]
        self.assertEqual(set(rows_by_corner), {(0, 0), (1, 0), (0, 1), (1, 1)})
        top_left = rows_by_corner[(0, 0)]
        bottom_right = rows_by_corner[(1, 1)]
        self.assertAlmostEqual(top_left[1], 0.875)  # (7 - 0) / 8
        self.assertAlmostEqual(top_left[3], 0.25)  # 2 / 8
        self.assertAlmostEqual(bottom_right[1], 0.125)  # (9 - 8) / 8

    def test_outside_and_degenerate_boxes_are_dropped(self) -> None:
        window = grid_windows(20, 20, 8, 8)[0]
        outside = YoloBox(class_id=1, x_min=12.0, y_min=12.0, x_max=16.0, y_max=16.0)
        sliver = YoloBox(class_id=1, x_min=6.0, y_min=2.0, x_max=6.5, y_max=6.0)
        rows, counts, degenerate = clip_boxes_to_window([outside, sliver], window)
        self.assertEqual(rows, [])
        self.assertEqual(counts, {})
        self.assertEqual(degenerate, 1)


class ReadYoloBoxesTests(unittest.TestCase):
    def test_reads_boxes_in_pixel_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "a.txt"
            label_path.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            boxes = read_yolo_boxes(label_path, image_width=100, image_height=50)
        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        self.assertEqual(box.class_id, 1)
        self.assertAlmostEqual(box.x_min, 40.0)
        self.assertAlmostEqual(box.y_min, 20.0)
        self.assertAlmostEqual(box.x_max, 60.0)
        self.assertAlmostEqual(box.y_max, 30.0)

    def test_malformed_lines_raise_with_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "a.txt"
            label_path.write_text("1 0.5 0.5 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"a\.txt:1"):
                read_yolo_boxes(label_path, image_width=100, image_height=100)


class EmptyChipSelectionTests(unittest.TestCase):
    def test_selection_count_rules(self) -> None:
        self.assertEqual(_select_empty_chip_count(10, 4, 0.15), 2)
        self.assertEqual(_select_empty_chip_count(10, 4, 0.0), 1)
        self.assertEqual(_select_empty_chip_count(10, 0, 0.5), 0)
        self.assertEqual(_select_empty_chip_count(10, 3, 1.0), 3)

    def test_plan_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "source"
            _write_source_dataset(source_root)
            kwargs = dict(
                source_root=source_root,
                output_root=Path(tmp_dir) / "out",
                tile_size=8,
                stride=6,
                empty_frac=0.3,
                seed=7,
            )
            first = build_tiling_plan(**kwargs)
            second = build_tiling_plan(**kwargs)
        self.assertEqual(
            [(chip.chip_stem, chip.rows) for chip in first.chips],
            [(chip.chip_stem, chip.rows) for chip in second.chips],
        )


class TilingEndToEndTests(unittest.TestCase):
    def _run(self, tmp_dir: str, extra_args: list[str]) -> tuple[Path, Path, int]:
        source_root = Path(tmp_dir) / "source"
        output_root = Path(tmp_dir) / "tiles"
        _write_source_dataset(source_root)
        before = _snapshot_checksums(source_root)
        exit_code = main(
            [
                "--source-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--tile-size",
                "8",
                "--stride",
                "6",
                "--empty-frac",
                "0.3",
                "--seed",
                "7",
                *extra_args,
            ]
        )
        self.assertEqual(_snapshot_checksums(source_root), before)
        return source_root, output_root, exit_code

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _source, output_root, exit_code = self._run(tmp_dir, ["--dry-run"])
            self.assertEqual(exit_code, 0)
            self.assertFalse(output_root.exists())

    def test_real_run_writes_tiled_dataset_and_keeps_source_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root, output_root, exit_code = self._run(tmp_dir, [])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "classes.txt").is_file())
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["tile_size"], 8)
            self.assertEqual(metadata["stride"], 6)
            self.assertTrue((output_root / "README.md").is_file())

            chips_per_split = metadata["splits"]
            for split, chip_count in chips_per_split.items():
                images = list((output_root / "images" / split).glob("*.jpg"))
                labels = list((output_root / "labels" / split).glob("*.txt"))
                self.assertEqual(len(images), chip_count)
                self.assertEqual(len(labels), chip_count)

            total_chips = metadata["total_chips"]
            labeled = 0
            empty = 0
            for split in chips_per_split:
                for label_path in (output_root / "labels" / split).glob("*.txt"):
                    stem, index_x, index_y = parse_chip_name(label_path.stem)
                    self.assertTrue((source_root / "images" / split / f"{stem}.jpg").is_file())
                    self.assertGreaterEqual(index_x, 0)
                    self.assertGreaterEqual(index_y, 0)
                    if label_path.read_text(encoding="utf-8").strip():
                        labeled += 1
                    else:
                        empty += 1
            self.assertEqual(labeled + empty, total_chips)
            self.assertEqual(metadata["empty_chips_kept"], empty)
            # Exact profile of the synthetic fixture: 4 source boxes; the central
            # 8x8px train box is duplicated onto all 9 chips; 11 clipped slivers
            # (1px side) across val/test are dropped as degenerate.
            self.assertEqual(metadata["source_boxes"], 4)
            self.assertEqual(metadata["kept_boxes"], 12)
            self.assertEqual(metadata["degenerate_clipped_boxes_dropped"], 11)
