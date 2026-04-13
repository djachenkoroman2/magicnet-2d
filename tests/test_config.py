from __future__ import annotations

from pathlib import Path
import sys
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

    def test_yolov3_tvdd_mmdet_config_loads(self) -> None:
        config = load_project_config(ROOT / "configs/project/yolov3_d53_tvdd_mmdet.toml")

        self.assertEqual(config.project.run_name, "yolov3_d53_tvdd_mmdet")
        self.assertEqual(config.mmdet.config_ref, "mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py")
        self.assertEqual(config.dataset.dataset_type, "YOLOTxtDataset")
        self.assertTrue(config.mmdet.disable_pretrained)
        self.assertEqual(config.train.num_workers, 0)
        self.assertEqual(len(config.dataset.classes), 23)
        self.assertEqual(
            config.dataset.train.ann_path(config.paths.data_root),
            ROOT / "data/tvdd/labels/train",
        )
        self.assertEqual(
            config.dataset.train.image_root(config.paths.data_root),
            ROOT / "data/tvdd/images/train",
        )


if __name__ == "__main__":
    unittest.main()
