from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.detection_metrics import (
    GroundTruthBox,
    PredictionBox,
    bbox_iou,
    compute_detection_metrics,
    match_predictions,
    write_evaluation_artifacts,
)


class DetectionMetricsTests(unittest.TestCase):
    def test_bbox_iou_computes_overlap_over_union(self) -> None:
        self.assertAlmostEqual(
            bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)),
            25 / 175,
        )

    def test_match_predictions_counts_tp_fp_and_fn(self) -> None:
        ground_truths = [
            GroundTruthBox("image-a", 0, (0, 0, 10, 10)),
            GroundTruthBox("image-a", 1, (20, 20, 30, 30)),
        ]
        predictions = [
            PredictionBox("image-a", 0, 0.9, (0, 0, 10, 10)),
            PredictionBox("image-a", 1, 0.8, (50, 50, 60, 60)),
        ]

        true_positives, false_positives, false_negatives = match_predictions(
            ground_truths,
            predictions,
            iou_thr=0.5,
            score_thr=0.1,
        )

        self.assertEqual(len(true_positives), 1)
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(len(false_negatives), 1)

    def test_compute_detection_metrics_reports_map_and_class_metrics(self) -> None:
        ground_truths = [
            GroundTruthBox("image-a", 0, (0, 0, 10, 10)),
            GroundTruthBox("image-a", 1, (20, 20, 30, 30)),
        ]
        predictions = [
            PredictionBox("image-a", 0, 0.9, (0, 0, 10, 10)),
            PredictionBox("image-a", 1, 0.8, (50, 50, 60, 60)),
        ]

        result = compute_detection_metrics(
            ground_truths,
            predictions,
            class_names=("substation", "supports 10"),
            iou_thrs=(0.5,),
            score_thr=0.1,
        )

        self.assertAlmostEqual(result.metrics["precision@0.5"], 0.5)
        self.assertAlmostEqual(result.metrics["recall@0.5"], 0.5)
        self.assertAlmostEqual(result.metrics["F1@0.5"], 0.5)
        self.assertAlmostEqual(result.metrics["mAP@0.5"], 0.5)
        self.assertAlmostEqual(result.metrics["AP@0.5/substation"], 1.0)
        self.assertAlmostEqual(result.metrics["AP@0.5/supports_10"], 0.0)

    def test_write_evaluation_artifacts_saves_debug_files(self) -> None:
        result = compute_detection_metrics(
            [GroundTruthBox("image-a", 0, (0, 0, 10, 10))],
            [PredictionBox("image-a", 0, 0.9, (0, 0, 10, 10))],
            class_names=("substation",),
            iou_thrs=(0.5,),
            score_thr=0.1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir) / "eval"
            write_evaluation_artifacts(artifact_dir, result, class_names=("substation",))

            self.assertTrue((artifact_dir / "metrics.json").exists())
            self.assertTrue((artifact_dir / "confusion_matrix.csv").exists())
            self.assertTrue((artifact_dir / "false_positives.json").exists())
            self.assertTrue((artifact_dir / "false_negatives.json").exists())


if __name__ == "__main__":
    unittest.main()
