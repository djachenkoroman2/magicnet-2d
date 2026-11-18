#!/usr/bin/env python3
"""Score-threshold sweep for a trained checkpoint on a validation split.

Runs the model over the val dataloader built by the project runtime config,
collects all predictions and ground truths once, then computes precision /
recall / F1 (overall and per class) at several score thresholds by re-matching
with the shared matching logic from magicnet_2d.detection_metrics.

Artifacts are written to outputs/evaluation/<run>/<split>_score_sweep/:
  sweep.csv  - one row per threshold with P/R/F1 overall and per class
  sweep.json - the same data plus run metadata
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from magicnet_2d.config import load_project_config  # noqa: E402
from magicnet_2d.detection_metrics import (  # noqa: E402
    GroundTruthBox,
    PredictionBox,
    _boxes_from_instances,
    match_predictions,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/project/energy_obj_tiles_cascade_ms3x_mmdet.toml",
        help="Path to the project TOML config.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint to evaluate.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6",
        help="Comma-separated score thresholds.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth.",
    )


def _prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _instances_to_dict(instances: Any, *, with_scores: bool) -> dict[str, Any]:
    """Unwrap mmengine/mmdet instance containers into a plain dict of tensors.

    gt/pred instances may carry HorizontalBoxes wrappers; detection_metrics
    helpers expect plain tensors (or dicts).
    """
    bboxes = instances.bboxes
    if hasattr(bboxes, "tensor"):
        bboxes = bboxes.tensor
    payload: dict[str, Any] = {"bboxes": bboxes, "labels": instances.labels}
    if with_scores:
        payload["scores"] = getattr(instances, "scores", None)
    return payload


def _class_counts(boxes) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for box in boxes:
        counts[box.label] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    args = parser.parse_args(argv)

    thresholds = [float(part) for part in args.thresholds.split(",") if part.strip()]
    if not thresholds:
        raise SystemExit("No thresholds given.")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    from mmengine.registry import MODELS
    from mmengine.runner import Runner, load_checkpoint
    import torch

    import mmdet.models  # noqa: F401 - registers mmdet models in the registry

    from mmengine.registry import DefaultScope

    from magicnet_2d.mmdet_support import _register_local_mmdet_modules

    project_config = load_project_config(args.config)
    from magicnet_2d.mmdet_support import build_runtime_config

    cfg, _, _ = build_runtime_config(project_config)
    DefaultScope.get_instance("eval_score_sweep", scope_name="mmdet")
    _register_local_mmdet_modules()

    if cfg.get("val_dataloader") is None:
        raise SystemExit("val_dataloader is disabled (train.validate=false); nothing to sweep.")

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint))
    model.eval()
    model.to("cuda:0")

    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    ground_truths: list[GroundTruthBox] = []
    predictions: list[PredictionBox] = []
    with torch.no_grad():
        for data_batch in dataloader:
            data_samples = data_batch["data_samples"]
            results = model.test_step(data_batch)
            for sample, result in zip(data_samples, results):
                image_id = str(sample.img_path)
                ground_truths.extend(
                    _boxes_from_instances(
                        _instances_to_dict(sample.gt_instances, with_scores=False),
                        image_id,
                        with_scores=False,
                    )
                )
                predictions.extend(
                    _boxes_from_instances(
                        _instances_to_dict(result.pred_instances, with_scores=True),
                        image_id,
                        with_scores=True,
                    )
                )

    class_names = list(project_config.dataset.classes)
    gt_per_class = _class_counts(ground_truths)
    rows: list[dict[str, object]] = []
    for thr in thresholds:
        tp_list, fp_list, fn_list = match_predictions(
            ground_truths, predictions, iou_thr=args.iou_thr, score_thr=thr
        )
        tp_c = _class_counts(tp_list)
        fp_c = _class_counts(fp_list)
        fn_c = _class_counts(fn_list)
        precision, recall, f1 = _prf_from_counts(len(tp_list), len(fp_list), len(fn_list))
        row: dict[str, object] = {
            "score_thr": thr,
            "tp": len(tp_list),
            "fp": len(fp_list),
            "fn": len(fn_list),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        for class_id, name in enumerate(class_names):
            c_tp = tp_c.get(class_id, 0)
            c_fp = fp_c.get(class_id, 0)
            c_fn = fn_c.get(class_id, 0)
            c_p, c_r, _ = _prf_from_counts(c_tp, c_fp, c_fn)
            row[f"{name}/precision"] = round(c_p, 4)
            row[f"{name}/recall"] = round(c_r, 4)
            row[f"{name}/gt"] = gt_per_class.get(class_id, 0)
        rows.append(row)

    artifact_dir = (
        project_config.paths.output_root
        / "evaluation"
        / project_config.project.run_name
        / "val_score_sweep"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (artifact_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (artifact_dir / "sweep.json").write_text(
        json.dumps(
            {
                "run_name": project_config.project.run_name,
                "checkpoint": str(checkpoint),
                "iou_thr": args.iou_thr,
                "num_val_images": len(dataloader.dataset),
                "classes": class_names,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    best_f1 = max(rows, key=lambda row: float(row["f1"]))
    print(f"Wrote {artifact_dir}/sweep.csv and sweep.json")
    print(
        "Best F1: score_thr={score_thr} -> P={precision} R={recall} F1={f1}".format(**best_f1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
