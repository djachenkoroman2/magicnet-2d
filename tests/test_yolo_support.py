from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.yolo_support import build_yolo_run_paths, copy_weight_artifacts


class YoloSupportTests(unittest.TestCase):
    def test_build_yolo_run_paths_uses_expected_directories(self) -> None:
        paths = build_yolo_run_paths(ROOT, "coco8_smoke")

        self.assertEqual(paths.log_project_dir, ROOT / "log/yolo")
        self.assertEqual(paths.run_dir, ROOT / "log/yolo/coco8_smoke")
        self.assertEqual(paths.checkpoint_dir, ROOT / "checkpoints/yolo/coco8_smoke")
        self.assertEqual(paths.output_project_dir, ROOT / "outputs/yolo")

    def test_copy_weight_artifacts_copies_best_and_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            source_dir = temp_root / "weights"
            target_dir = temp_root / "checkpoints"
            source_dir.mkdir(parents=True)
            (source_dir / "best.pt").write_text("best", encoding="utf-8")
            (source_dir / "last.pt").write_text("last", encoding="utf-8")

            copied = copy_weight_artifacts(source_dir, target_dir)

            self.assertEqual(sorted(path.name for path in copied), ["best.pt", "last.pt"])
            self.assertEqual((target_dir / "best.pt").read_text(encoding="utf-8"), "best")
            self.assertEqual((target_dir / "last.pt").read_text(encoding="utf-8"), "last")


if __name__ == "__main__":
    unittest.main()
