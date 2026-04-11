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


if __name__ == "__main__":
    unittest.main()
