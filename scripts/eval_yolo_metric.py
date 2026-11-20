#!/usr/bin/env python3
"""Evaluate a trained Ultralytics YOLO checkpoint with the project YOLOTxtMetric.

Runs inference over a YOLO-txt val split and computes mAP@0.5 / mAP@0.5:0.95,
per-class AP and P/R/F1 using the exact same matching code as the MMDetection
pipeline (magicnet_2d.detection_metrics). This makes YOLO numbers directly
comparable to the mmdet runs (DINO / cascade / FRCNN) instead of relying on the
looser Ultralytics built-in metric (conf=0.001, 101-pt interpolation).

Artifacts are written to --output-dir (metrics.json + confusion matrix + FP/FN).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.config import load_project_config  # noqa: E402
from magicnet_2d.detection_metrics import (  # noqa: E402
    GroundTruthBox,
    PredictionBox,
    compute_detection_metrics,
    write_evaluation_artifacts,
    _sanitize_metric_name,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained Ultralytics .pt checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/project/energy_obj_tiles_cascade_ms3x_mmdet.toml",
        help="Project TOML config (used for classes and val-split paths).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size (must match the training imgsz).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence floor for collecting predictions (keep near 0 to mirror mmdet AP).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per image.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for P/R/F1 (matches the mmdet YOLOTxtMetric setting).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation artifacts (defaults to outputs/evaluation/<run>/val).",
    )


def _parse_yolo_label(label_path: Path, width: int, height: int) -> list[GroundTruthBox]:
    boxes: list[GroundTruthBox] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:])
        if w <= 0 or h <= 0:
            continue
        x1 = max(0.0, (xc - w / 2.0) * width)
        y1 = max(0.0, (yc - h / 2.0) * height)
        x2 = min(width, (xc + w / 2.0) * width)
        y2 = min(height, (yc + h / 2.0) * height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(GroundTruthBox("", cls, (x1, y1, x2, y2)))
    return boxes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    args = parser.parse_args(argv)

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")

    from ultralytics import YOLO

    project_config = load_project_config(args.config)
    class_names = list(project_config.dataset.classes)
    data_root = project_config.paths.data_root
    images_dir = project_config.dataset.val.image_root(data_root)
    labels_dir = project_config.dataset.val.ann_path(data_root)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_config.paths.output_root
        / "evaluation"
        / weights.parent.name
        / "val"
    )

    model = YOLO(str(weights))

    ground_truths: list[GroundTruthBox] = []
    predictions: list[PredictionBox] = []

    for result in model.predict(
        source=str(images_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        save=False,
    ):
        image_path = Path(result.path)
        image_id = image_path.name
        height, width = result.orig_shape[:2]

        label_path = labels_dir / image_path.with_suffix(".txt").name
        for box in _parse_yolo_label(label_path, width, height):
            ground_truths.append(GroundTruthBox(image_id, box.label, box.bbox))

        if result.boxes is None:
            continue
        for xyxy, cls, conf in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            predictions.append(
                PredictionBox(image_id, int(cls), float(conf), tuple(float(v) for v in xyxy))
            )

    eval_result = compute_detection_metrics(
        ground_truths,
        predictions,
        class_names=class_names,
        score_thr=args.score_thr,
    )
    write_evaluation_artifacts(output_dir, eval_result, class_names=class_names)

    m = eval_result.metrics
    print(f"Checkpoint: {weights.name}")
    print(f"mAP@0.5={m['mAP@0.5']:.4f}  mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}  "
          f"P={m['precision@0.5']:.3f}  R={m['recall@0.5']:.3f}  F1={m['F1@0.5']:.3f}")
    for class_id, name in enumerate(class_names):
        ap50 = m.get(f"AP@0.5/{_sanitize_metric_name(name)}", 0.0)
        print(f"  {name:15s} AP@0.5={ap50:.4f}")
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
