from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.mmdet_support import MissingMMDetectionRuntimeError, _missing_runtime_message
from magicnet_2d.train import run_from_args


def _write_project_config(tmp_root: Path) -> Path:
    config_path = tmp_root / "project.toml"
    config_path.write_text(
        """
[project]
name = "test-project"
run_name = "test-run"

[paths]
data_root = "data/coco"
log_root = "log"
checkpoint_root = "checkpoints"
output_root = "outputs"

[mmdet]
config = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
checkpoint = ""

[dataset]
type = "CocoDataset"
classes = ["object"]

[dataset.train]
ann_file = "annotations/instances_train.json"
img_prefix = "train/"

[dataset.val]
ann_file = "annotations/instances_val.json"
img_prefix = "val/"

[dataset.test]
ann_file = "annotations/instances_val.json"
img_prefix = "val/"

[train]
seed = 42
validate = true
amp = false
max_epochs = 1
checkpoint_interval = 1
max_keep_checkpoints = 1
logger_interval = 10
resume = ""

[inference]
device = "auto"
score_threshold = 0.3
batch_size = 1
output_subdir = "inference"
""".strip(),
        encoding="utf-8",
    )
    return config_path


class MMDetSupportTests(unittest.TestCase):
    def test_build_runtime_config_disables_validation_components_together(self) -> None:
        from magicnet_2d.config import load_project_config
        from magicnet_2d.mmdet_support import build_runtime_config

        class DummyConfig(dict):
            @classmethod
            def fromfile(cls, _: str) -> "DummyConfig":
                return cls(
                    train_cfg={"max_epochs": 12, "val_interval": 1},
                    train_dataloader={"dataset": {"type": "CocoDataset", "ann_file": "train.json", "data_prefix": {}}},
                    val_dataloader={"dataset": {"type": "CocoDataset", "ann_file": "val.json", "data_prefix": {}}},
                    test_dataloader={"dataset": {"type": "CocoDataset", "ann_file": "test.json", "data_prefix": {}}},
                    val_evaluator={"ann_file": "val.json"},
                    test_evaluator={"ann_file": "test.json"},
                    model={"bbox_head": {"num_classes": 80}},
                    default_hooks={},
                    visualizer={},
                )

            def __getattr__(self, name: str):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name: str, value) -> None:
                self[name] = value

        project_config = load_project_config(ROOT / "configs/project/yolov3_d53_tvdd_mmdet.toml")

        with patch("magicnet_2d.mmdet_support.resolve_mmdet_config", return_value=ROOT / "dummy.py"):
            with patch("magicnet_2d.mmdet_support._require_mmdet_training_runtime", return_value=(DummyConfig, object())):
                with patch("magicnet_2d.mmdet_support._register_local_mmdet_modules"):
                    with patch("magicnet_2d.mmdet_support._prepare_mmdet_environment"):
                        cfg, _, _ = build_runtime_config(project_config)

        self.assertIsNone(cfg.val_dataloader)
        self.assertIsNone(cfg.val_cfg)
        self.assertIsNone(cfg.val_evaluator)
        self.assertIsNotNone(cfg.test_dataloader)
        self.assertEqual(cfg.train_dataloader["num_workers"], 0)
        self.assertFalse(cfg.train_dataloader["persistent_workers"])

    def test_build_runtime_config_can_disable_pretrained_init(self) -> None:
        from magicnet_2d.config import load_project_config
        from magicnet_2d.mmdet_support import build_runtime_config

        class DummyConfig(dict):
            @classmethod
            def fromfile(cls, _: str) -> "DummyConfig":
                return cls(
                    train_cfg={"max_epochs": 12},
                    train_dataloader={"dataset": {"type": "CocoDataset", "ann_file": "train.json", "data_prefix": {}}},
                    val_dataloader={"dataset": {"type": "CocoDataset", "ann_file": "val.json", "data_prefix": {}}},
                    test_dataloader={"dataset": {"type": "CocoDataset", "ann_file": "test.json", "data_prefix": {}}},
                    val_evaluator={"ann_file": "val.json"},
                    test_evaluator={"ann_file": "test.json"},
                    model={
                        "backbone": {"init_cfg": {"type": "Pretrained", "checkpoint": "open-mmlab://darknet53"}},
                        "neck": {"pretrained": "torchvision://resnet50"},
                        "bbox_head": {"num_classes": 80},
                    },
                    default_hooks={},
                    visualizer={},
                )

            def __getattr__(self, name: str):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name: str, value) -> None:
                self[name] = value

        project_config = load_project_config(ROOT / "configs/project/yolov3_d53_tvdd_mmdet.toml")

        with patch("magicnet_2d.mmdet_support.resolve_mmdet_config", return_value=ROOT / "dummy.py"):
            with patch("magicnet_2d.mmdet_support._require_mmdet_training_runtime", return_value=(DummyConfig, object())):
                with patch("magicnet_2d.mmdet_support._register_local_mmdet_modules"):
                    with patch("magicnet_2d.mmdet_support._prepare_mmdet_environment"):
                        cfg, _, _ = build_runtime_config(project_config)

        self.assertIsNone(cfg.model["backbone"]["init_cfg"])
        self.assertIsNone(cfg.model["neck"]["pretrained"])

    def test_missing_runtime_message_mentions_uv_run_and_mmdet_config_refs(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name == "mmengine":
                return object()
            if name == "mmdet":
                return None
            raise AssertionError(f"Unexpected package lookup: {name}")

        with patch("magicnet_2d.mmdet_support.find_spec", side_effect=fake_find_spec):
            message = _missing_runtime_message("mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py")

        self.assertIn("Missing Python packages: `mmdet`.", message)
        self.assertIn("must be available even for `--dry-run`", message)
        self.assertIn("Re-run the command with `uv run ...`", message)

    def test_train_dry_run_returns_one_with_human_readable_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_project_config(Path(tmp_dir))
            args = argparse.Namespace(
                config=str(config_path),
                mmdet_config=None,
                checkpoint=None,
                run_name=None,
                dry_run=True,
            )
            stdout = io.StringIO()
            error = MissingMMDetectionRuntimeError("MMDetection runtime is not installed.\nMissing Python packages: `mmdet`.")

            with patch("magicnet_2d.train.build_runtime_config", side_effect=error):
                with redirect_stdout(stdout):
                    exit_code = run_from_args(args)

        self.assertEqual(exit_code, 1)
        output = stdout.getvalue()
        self.assertIn("Training dataset paths are missing:", output)
        self.assertIn("MMDetection runtime is not installed.", output)
        self.assertIn("Missing Python packages: `mmdet`.", output)


if __name__ == "__main__":
    unittest.main()
