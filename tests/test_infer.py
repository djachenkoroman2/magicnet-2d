from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.infer import _merge_inferencer_summaries, _summarize_inferencer_results


class FakeArray:
    shape = (1024, 2048, 3)
    dtype = "uint8"

    def tolist(self) -> list[int]:
        raise AssertionError("visualization arrays should not be expanded into JSON")


class InferTests(unittest.TestCase):
    def test_summarize_inferencer_results_keeps_visualizations_compact(self) -> None:
        summary = _summarize_inferencer_results(
            {
                "predictions": [{"labels": [], "scores": [], "bboxes": []}],
                "visualization": [FakeArray()],
            }
        )

        self.assertEqual(summary["predictions"], [{"labels": [], "scores": [], "bboxes": []}])
        self.assertEqual(
            summary["visualization"],
            [{"type": "FakeArray", "shape": [1024, 2048, 3], "dtype": "uint8"}],
        )

    def test_merge_inferencer_summaries_concatenates_batch_lists(self) -> None:
        merged = _merge_inferencer_summaries(
            [
                {"predictions": [{"labels": [1]}], "visualization": [{"shape": [1, 2, 3]}]},
                {"predictions": [{"labels": [2]}], "visualization": [{"shape": [4, 5, 6]}]},
            ]
        )

        self.assertEqual(merged["predictions"], [{"labels": [1]}, {"labels": [2]}])
        self.assertEqual(merged["visualization"], [{"shape": [1, 2, 3]}, {"shape": [4, 5, 6]}])


if __name__ == "__main__":
    unittest.main()
