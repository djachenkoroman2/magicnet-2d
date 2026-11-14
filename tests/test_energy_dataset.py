from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.energy_dataset import (
    build_class_config,
    build_energy_dataset_plan,
    discover_energy_items,
    normalize_label_stem,
    prepare_energy_dataset,
    prune_empty_label_items,
    split_items,
)


def _write_notes(project_dir: Path, categories: dict[int, str]) -> None:
    (project_dir / "notes.json").write_text(
        json.dumps(
            {
                "categories": [{"id": class_id, "name": name} for class_id, name in categories.items()],
                "info": {"contributor": "unit-test"},
            }
        ),
        encoding="utf-8",
    )


class EnergyDatasetTests(unittest.TestCase):
    def test_normalize_label_stem_handles_uuid_prefix(self) -> None:
        image_stems = {"Tile-0-0", "Tile-0-1"}

        self.assertEqual(normalize_label_stem("deadbeef-Tile-0-0", image_stems), "Tile-0-0")
        self.assertEqual(normalize_label_stem("Tile-0-1", image_stems), "Tile-0-1")
        self.assertIsNone(normalize_label_stem("deadbeef-Other", image_stems))

    def test_discover_matches_prefixed_labels_and_geo_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "energy2"
            project_dir = source_root / "project-a"
            images_dir = project_dir / "images"
            labels_dir = project_dir / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir()
            _write_notes(project_dir, {0: "substation", 1: "support"})

            image_path = images_dir / "Tile-0-0.jpg"
            image_path.write_bytes(b"fake")
            image_path.with_suffix(".jgw").write_text("1\n0\n0\n-1\n0\n0\n", encoding="utf-8")
            (labels_dir / "deadbeef-Tile-0-0.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            items, categories, class_counts, unmatched_labels, conflicts = discover_energy_items(source_root)

        self.assertEqual(conflicts, ())
        self.assertEqual(categories, {0: "substation", 1: "support"})
        self.assertEqual(class_counts[1], 1)
        self.assertEqual(unmatched_labels, ())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].label_path.name, "deadbeef-Tile-0-0.txt")
        self.assertEqual(items[0].relative_label_path.as_posix(), "Tile-0-0.txt")
        self.assertEqual([path.suffix for path in items[0].geo_paths], [".jgw"])

    def test_normalize_label_stem_handles_percent_encoded_names(self) -> None:
        image_stems = {"35кВ_1-0-8", "Tile-0-0"}

        self.assertEqual(normalize_label_stem("35%D0%BA%D0%92_1-0-8", image_stems), "35кВ_1-0-8")
        self.assertEqual(normalize_label_stem("abcd1234-35%D0%BA%D0%92_1-0-8", image_stems), "35кВ_1-0-8")
        self.assertEqual(normalize_label_stem("deadbeef-Tile-0-0", image_stems), "Tile-0-0")

    def test_discover_resolves_duplicate_labels_by_annotation_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "energy2"
            project_dir = source_root / "project-a"
            images_dir = project_dir / "images"
            labels_dir = project_dir / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir()
            _write_notes(project_dir, {0: "substation", 1: "support"})

            image_path = images_dir / "Tile-0-0.jpg"
            image_path.write_bytes(b"fake")
            (labels_dir / "aaaa1111-Tile-0-0.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            richer_label = labels_dir / "bbbb2222-Tile-0-0.txt"
            richer_label.write_text(
                "1 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n",
                encoding="utf-8",
            )

            items, _, class_counts, unmatched_labels, conflicts = discover_energy_items(source_root)

            self.assertEqual(unmatched_labels, ())
            self.assertEqual(class_counts[1], 2)
            self.assertEqual(len(conflicts), 1)
            conflict = conflicts[0]
            self.assertEqual(conflict.image_stem, "Tile-0-0")
            self.assertEqual(conflict.kept_label_path, richer_label)
            self.assertEqual(len(conflict.rejected_label_paths), 1)
            self.assertEqual(items[0].label_path, richer_label)

    def test_prepare_writes_normalized_layout_and_preserves_empty_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / "energy2"
            output_root = root / "energy_obj"
            project_dir = source_root / "project-a"
            images_dir = project_dir / "images"
            labels_dir = project_dir / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir()
            _write_notes(project_dir, {0: "substation", 1: "support"})

            first_image = images_dir / "Tile-0-0.jpg"
            first_image.write_bytes(b"fake-1")
            first_image.with_suffix(".jgw").write_text("world", encoding="utf-8")
            first_image.with_suffix(".prj").write_text("projection", encoding="utf-8")
            (labels_dir / "deadbeef-Tile-0-0.txt").write_text("", encoding="utf-8")

            second_image = images_dir / "Tile-0-1.jpg"
            second_image.write_bytes(b"fake-2")
            (labels_dir / "cafebabe-Tile-0-1.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            plan = build_energy_dataset_plan(
                source_root=source_root,
                output_root=output_root,
                train_ratio=1.0,
                val_ratio=0.0,
                seed=42,
            )
            prepare_energy_dataset(plan, dry_run=False, force=False)

            normalized_empty_label = output_root / "labels/train/Tile-0-0.txt"
            normalized_object_label = output_root / "labels/train/Tile-0-1.txt"
            geo_world = output_root / "geo/train/Tile-0-0.jgw"
            geo_projection = output_root / "geo/train/Tile-0-0.prj"

            self.assertTrue((output_root / "images/train/Tile-0-0.jpg").exists())
            self.assertEqual(normalized_empty_label.read_text(encoding="utf-8"), "")
            self.assertIn("0.5 0.5", normalized_object_label.read_text(encoding="utf-8"))
            self.assertTrue(geo_world.exists())
            self.assertTrue(geo_projection.exists())
            self.assertTrue((output_root / "README.md").exists())
            self.assertTrue((output_root / "metadata.json").exists())

    def test_discover_rejects_flattened_name_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "energy2"
            for project_name in ("project-a", "project-b"):
                project_dir = source_root / project_name
                images_dir = project_dir / "images"
                images_dir.mkdir(parents=True)
                _write_notes(project_dir, {0: "substation"})
                (images_dir / "Tile-0-0.jpg").write_bytes(b"fake")

            with self.assertRaisesRegex(ValueError, "would overwrite files with the same name"):
                discover_energy_items(source_root)

    def test_sparse_class_config_uses_class_id_map(self) -> None:
        class_config = build_class_config({2: "support 10", 4: "transformer"}, {2, 4})

        self.assertEqual(class_config.classes, ("support 10", "transformer"))
        self.assertEqual(class_config.class_id_map, (2, 4))

    def test_contiguous_class_config_does_not_need_class_id_map(self) -> None:
        class_config = build_class_config({0: "substation", 1: "support"}, {1})

        self.assertEqual(class_config.classes, ("substation", "support"))
        self.assertIsNone(class_config.class_id_map)

    def test_prune_dry_run_moves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / "energy2"
            output_root = root / "energy_obj"
            project_dir = source_root / "project-a"
            images_dir = project_dir / "images"
            labels_dir = project_dir / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir()
            _write_notes(project_dir, {0: "substation", 1: "support"})

            image_path = images_dir / "Tile-0-0.jpg"
            image_path.write_bytes(b"fake")
            (labels_dir / "deadbeef-Tile-0-0.txt").write_text("", encoding="utf-8")

            plan = build_energy_dataset_plan(
                source_root=source_root,
                output_root=output_root,
                train_ratio=1.0,
                val_ratio=0.0,
                seed=42,
            )
            prepare_energy_dataset(plan, dry_run=False, force=False)

            holding_root = root / "unlabeled"
            summary = prune_empty_label_items(
                dataset_root=output_root, output_root=holding_root, dry_run=True
            )

            self.assertEqual(summary["pruned_images"], 1)
            self.assertFalse(holding_root.exists())
            self.assertTrue((output_root / "images" / "train" / "Tile-0-0.jpg").exists())

    def test_prune_moves_empty_label_items_and_updates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / "energy2"
            output_root = root / "energy_obj"
            project_dir = source_root / "project-a"
            images_dir = project_dir / "images"
            labels_dir = project_dir / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir()
            _write_notes(project_dir, {0: "substation", 1: "support"})

            labeled_image = images_dir / "Tile-0-0.jpg"
            labeled_image.write_bytes(b"fake-1")
            labeled_image.with_suffix(".jgw").write_text("world", encoding="utf-8")
            (labels_dir / "deadbeef-Tile-0-0.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            unlabeled_image = images_dir / "Tile-0-1.jpg"
            unlabeled_image.write_bytes(b"fake-2")
            (labels_dir / "cafebabe-Tile-0-1.txt").write_text("", encoding="utf-8")

            plan = build_energy_dataset_plan(
                source_root=source_root,
                output_root=output_root,
                train_ratio=1.0,
                val_ratio=0.0,
                seed=42,
            )
            prepare_energy_dataset(plan, dry_run=False, force=False)

            holding_root = root / "unlabeled"
            summary = prune_empty_label_items(
                dataset_root=output_root, output_root=holding_root, dry_run=False
            )

            self.assertEqual(summary["pruned_images"], 1)
            self.assertEqual(summary["pruned_by_split"], {"train": 1})

            self.assertTrue((holding_root / "images" / "Tile-0-1.jpg").exists())
            self.assertTrue((holding_root / "labels" / "Tile-0-1.txt").exists())
            manifest = json.loads((holding_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["items"]["Tile-0-1.jpg"]["split"], "train")

            self.assertTrue((output_root / "images" / "train" / "Tile-0-0.jpg").exists())
            self.assertTrue((output_root / "geo" / "train" / "Tile-0-0.jgw").exists())
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["total_images"], 1)
            self.assertEqual(metadata["total_labels"], 1)
            self.assertEqual(metadata["empty_labels"], 0)
            self.assertEqual(metadata["splits"], {"train": 1, "val": 0, "test": 0})
            self.assertEqual(metadata["pruned_unlabeled"]["images"], 1)
            self.assertEqual(metadata["pruned_unlabeled"]["root"], str(holding_root.resolve()))

            readme = (output_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("`train`: 1 images", readme)
            self.assertIn("## Pruned Unlabeled Images", readme)
            self.assertIn(str(holding_root.resolve()), readme)
            self.assertNotIn("Tile-0-1", readme)

            again = prune_empty_label_items(
                dataset_root=output_root, output_root=holding_root, dry_run=False
            )
            self.assertEqual(again["pruned_images"], 0)

    def test_split_items_is_deterministic_and_complete(self) -> None:
        first = split_items(list(range(10)), train_ratio=0.6, val_ratio=0.2, seed=7)
        second = split_items(list(range(10)), train_ratio=0.6, val_ratio=0.2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual({name: len(items) for name, items in first.items()}, {"train": 6, "val": 2, "test": 2})
        self.assertEqual(sorted(item for items in first.values() for item in items), list(range(10)))


if __name__ == "__main__":
    unittest.main()
