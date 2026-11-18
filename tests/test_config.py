from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.config import load_project_config, override_config


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/default.toml")

        self.assertEqual(config.project.run_name, "faster_rcnn_r50_demo")
        self.assertEqual(config.paths.log_root, ROOT / "log")
        self.assertEqual(
            config.dataset.train.ann_path(config.paths.data_root),
            ROOT / "data/coco/annotations/instances_train.json",
        )

    def test_runtime_overrides_apply(self) -> None:
        config = load_project_config(ROOT / "configs/project/default.toml")
        overridden = override_config(config, run_name="unit-test-run", score_threshold=0.65)

        self.assertEqual(overridden.project.run_name, "unit-test-run")
        self.assertEqual(overridden.inference.score_threshold, 0.65)

    def test_filter_empty_gt_defaults_to_none(self) -> None:
        config = load_project_config(ROOT / "configs/project/default.toml")

        self.assertIsNone(config.dataset.filter_empty_gt)

    def test_filter_empty_gt_parses_from_toml(self) -> None:
        toml_text = """
[project]
name = "unit-test"
run_name = "unit-test"

[paths]
data_root = "data/unit"
log_root = "log"
checkpoint_root = "checkpoints"
output_root = "outputs"

[mmdet]
config = "mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py"

[dataset]
type = "YOLOTxtDataset"
filter_empty_gt = false
classes = ["a", "b"]

[dataset.train]
ann_file = "labels/train"
img_prefix = "images/train"

[dataset.val]
ann_file = "labels/val"
img_prefix = "images/val"

[dataset.test]
ann_file = "labels/test"
img_prefix = "images/test"

[train]
max_epochs = 1

[inference]
score_threshold = 0.3
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "unit.toml"
            config_path.write_text(toml_text, encoding="utf-8")
            config = load_project_config(config_path)

        self.assertFalse(config.dataset.filter_empty_gt)

    def test_filter_empty_gt_rejects_non_boolean(self) -> None:
        toml_text = """
[project]
name = "unit-test"
run_name = "unit-test"

[paths]
data_root = "data/unit"
log_root = "log"
checkpoint_root = "checkpoints"
output_root = "outputs"

[mmdet]
config = "mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py"

[dataset]
type = "YOLOTxtDataset"
filter_empty_gt = "false"
classes = ["a", "b"]

[dataset.train]
ann_file = "labels/train"
img_prefix = "images/train"

[dataset.val]
ann_file = "labels/val"
img_prefix = "images/val"

[dataset.test]
ann_file = "labels/test"
img_prefix = "images/test"

[train]
max_epochs = 1

[inference]
score_threshold = 0.3
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "unit.toml"
            config_path.write_text(toml_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset.filter_empty_gt"):
                load_project_config(config_path)

    def test_retinanet_coco_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/retinanet_r50_fpn_1x_coco.toml")

        self.assertEqual(config.project.run_name, "retinanet_r50_fpn_1x_coco")
        self.assertEqual(config.mmdet.config_ref, "mmdet::retinanet/retinanet_r50_fpn_1x_coco.py")
        self.assertFalse(config.mmdet.disable_pretrained)
        self.assertIsNone(config.train.num_workers)
        self.assertEqual(len(config.dataset.classes), 80)
        self.assertEqual(
            config.dataset.train.ann_path(config.paths.data_root),
            ROOT / "data/coco/annotations/instances_train2014.json",
        )
        self.assertEqual(
            config.dataset.val.image_root(config.paths.data_root),
            ROOT / "data/coco/val2014",
        )
        self.assertIsNone(config.dataset.class_id_map)

    def test_yolov3_tvdd_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/yolov3_d53_tvdd_mmdet.toml")

        self.assertEqual(config.project.run_name, "yolov3_d53_tvdd_mmdet")
        self.assertEqual(config.mmdet.config_ref, "mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py")
        self.assertEqual(config.dataset.dataset_type, "YOLOTxtDataset")
        self.assertTrue(config.mmdet.disable_pretrained)
        self.assertIsNone(config.train.num_workers)
        self.assertIsNone(config.dataset.class_id_map)
        self.assertEqual(len(config.dataset.classes), 23)
        self.assertEqual(
            config.dataset.train.ann_path(config.paths.data_root),
            ROOT / "data/tvdd/labels/train",
        )
        self.assertEqual(
            config.dataset.train.image_root(config.paths.data_root),
            ROOT / "data/tvdd/images/train",
        )

    def test_energy_obj_tiles_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/energy_obj_tiles_mmdet.toml")

        self.assertEqual(config.project.run_name, "energy_obj_tiles_mmdet")
        self.assertEqual(config.mmdet.config_ref, "mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py")
        self.assertEqual(config.dataset.dataset_type, "YOLOTxtDataset")
        self.assertFalse(config.mmdet.disable_pretrained)
        self.assertIsNone(config.dataset.class_id_map)
        self.assertFalse(config.dataset.filter_empty_gt)
        self.assertEqual(config.train.num_workers, 8)
        self.assertEqual(config.train.max_epochs, 100)
        self.assertEqual(
            config.dataset.classes,
            ("substation", "supports 0,4", "supports 10", "supports 35", "transformer"),
        )
        self.assertEqual(
            config.dataset.train.ann_path(config.paths.data_root),
            ROOT / "data/energy_obj_tiles/labels/train",
        )
        self.assertEqual(
            config.dataset.test.image_root(config.paths.data_root),
            ROOT / "data/energy_obj_tiles/images/test",
        )
    def test_energy_obj_tiles_frcnn_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/energy_obj_tiles_frcnn_mmdet.toml")

        self.assertEqual(config.project.run_name, "energy_obj_tiles_frcnn_mmdet")
        self.assertEqual(
            config.mmdet.config_ref,
            "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
        )
        self.assertEqual(
            config.mmdet.checkpoint,
            ROOT / "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        )
        self.assertFalse(config.mmdet.disable_pretrained)
        self.assertEqual(config.dataset.dataset_type, "YOLOTxtDataset")
        self.assertFalse(config.dataset.filter_empty_gt)
        self.assertEqual(config.train.num_workers, 8)
        self.assertEqual(config.train.max_epochs, 100)
        self.assertEqual(
            config.dataset.train.ann_path(config.paths.data_root),
            ROOT / "data/energy_obj_tiles/labels/train",
        )
    def test_energy_obj_tiles_yolo273_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/energy_obj_tiles_yolo273_mmdet.toml")

        self.assertEqual(config.project.run_name, "energy_obj_tiles_yolo273")
        self.assertEqual(config.mmdet.config_ref, "mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py")
        self.assertEqual(config.dataset.dataset_type, "YOLOTxtDataset")
        self.assertFalse(config.dataset.filter_empty_gt)
        self.assertEqual(config.train.max_epochs, 273)
        self.assertEqual(config.train.num_workers, 8)
    def test_energy_obj_tiles_frcnn_ms3x_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/energy_obj_tiles_frcnn_ms3x_mmdet.toml")

        self.assertEqual(config.project.run_name, "energy_obj_tiles_frcnn_ms3x")
        self.assertEqual(
            config.mmdet.config_ref,
            "mmdet::faster_rcnn/faster-rcnn_r50_fpn_ms-3x_coco.py",
        )
        self.assertEqual(
            config.mmdet.checkpoint,
            ROOT / "checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth",
        )
        self.assertEqual(config.train.max_epochs, 12)
        self.assertEqual(config.train.num_workers, 8)
    def test_energy_obj_tiles_cascade_ms3x_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/energy_obj_tiles_cascade_ms3x_mmdet.toml")

        self.assertEqual(config.project.run_name, "energy_obj_tiles_cascade_ms3x")
        self.assertEqual(
            config.mmdet.config_ref,
            "configs/mmdet/cascade-rcnn_r50_fpn_ms3x_energy.py",
        )
        self.assertEqual(
            config.mmdet.checkpoint,
            ROOT / "checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth",
        )
        self.assertEqual(config.train.max_epochs, 12)
        self.assertEqual(config.train.num_workers, 8)


if __name__ == "__main__":
    unittest.main()
