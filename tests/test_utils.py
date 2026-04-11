from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.config import load_project_config
from magicnet_2d.utils import build_run_directories, list_image_inputs


class UtilsTests(unittest.TestCase):
    def test_build_run_directories_uses_log_folder_for_tensorboard(self) -> None:
        config = load_project_config(ROOT / "configs/project/default.toml")
        run_dirs = build_run_directories(config.paths, "demo-run", "inference")

        self.assertEqual(run_dirs.run_dir, ROOT / "log/demo-run")
        self.assertEqual(run_dirs.tensorboard_dir, ROOT / "log/tensorboard/demo-run")
        self.assertEqual(run_dirs.checkpoint_dir, ROOT / "checkpoints/demo-run")

    def test_list_image_inputs_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            (temp_root / "a.jpg").write_text("stub", encoding="utf-8")
            (temp_root / "b.png").write_text("stub", encoding="utf-8")
            (temp_root / "notes.txt").write_text("ignore", encoding="utf-8")

            items = list_image_inputs(temp_root)

            self.assertEqual([path.name for path in items], ["a.jpg", "b.png"])


if __name__ == "__main__":
    unittest.main()
