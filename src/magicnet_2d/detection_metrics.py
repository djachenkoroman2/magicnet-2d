from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: str
    label: int
    bbox: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class PredictionBox:
    image_id: str
    label: int
    score: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class MatchedPrediction:
    prediction: PredictionBox
    ground_truth: GroundTruthBox
    iou: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    false_positives: list[dict[str, Any]]
    false_negatives: list[dict[str, Any]]


def _sanitize_metric_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", name.strip())
    return sanitized.strip("_") or "class"


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _mean(values: Iterable[float]) -> float:
    collected = [value for value in values if not math.isnan(value)]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _group_ground_truths_by_image(
    ground_truths: Sequence[GroundTruthBox],
    *,
    label: int | None = None,
) -> dict[str, list[GroundTruthBox]]:
    grouped: dict[str, list[GroundTruthBox]] = {}
    for ground_truth in ground_truths:
        if label is not None and ground_truth.label != label:
            continue
        grouped.setdefault(ground_truth.image_id, []).append(ground_truth)
    return grouped


def average_precision(
    ground_truths: Sequence[GroundTruthBox],
    predictions: Sequence[PredictionBox],
    *,
    class_id: int,
    iou_thr: float,
) -> float | None:
    class_ground_truths = [ground_truth for ground_truth in ground_truths if ground_truth.label == class_id]
    if not class_ground_truths:
        return None

    class_predictions = sorted(
        (prediction for prediction in predictions if prediction.label == class_id),
        key=lambda prediction: prediction.score,
        reverse=True,
    )
    if not class_predictions:
        return 0.0

    gt_by_image = _group_ground_truths_by_image(class_ground_truths)
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    true_positives: list[float] = []
    false_positives: list[float] = []

    for prediction in class_predictions:
        image_ground_truths = gt_by_image.get(prediction.image_id, [])
        best_iou = 0.0
        best_index: int | None = None
        for index, ground_truth in enumerate(image_ground_truths):
            if index in matched[prediction.image_id]:
                continue
            iou = bbox_iou(prediction.bbox, ground_truth.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_thr:
            matched[prediction.image_id].add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    cumulative_tp: list[float] = []
    cumulative_fp: list[float] = []
    tp_total = 0.0
    fp_total = 0.0
    for tp_value, fp_value in zip(true_positives, false_positives):
        tp_total += tp_value
        fp_total += fp_value
        cumulative_tp.append(tp_total)
        cumulative_fp.append(fp_total)

    recalls = [tp / len(class_ground_truths) for tp in cumulative_tp]
    precisions = [
        _safe_divide(tp, tp + fp)
        for tp, fp in zip(cumulative_tp, cumulative_fp)
    ]

    ap_values: list[float] = []
    for recall_level in [index / 100.0 for index in range(101)]:
        precision_at_recall = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= recall_level
        ]
        ap_values.append(max(precision_at_recall) if precision_at_recall else 0.0)
    return sum(ap_values) / len(ap_values)


def match_predictions(
    ground_truths: Sequence[GroundTruthBox],
    predictions: Sequence[PredictionBox],
    *,
    iou_thr: float,
    score_thr: float,
) -> tuple[list[MatchedPrediction], list[PredictionBox], list[GroundTruthBox]]:
    filtered_predictions = sorted(
        (prediction for prediction in predictions if prediction.score >= score_thr),
        key=lambda prediction: prediction.score,
        reverse=True,
    )
    gt_by_image_and_label: dict[tuple[str, int], list[GroundTruthBox]] = {}
    for ground_truth in ground_truths:
        gt_by_image_and_label.setdefault((ground_truth.image_id, ground_truth.label), []).append(ground_truth)

    matched_indices: dict[tuple[str, int], set[int]] = {
        key: set() for key in gt_by_image_and_label
    }
    true_positives: list[MatchedPrediction] = []
    false_positives: list[PredictionBox] = []

    for prediction in filtered_predictions:
        key = (prediction.image_id, prediction.label)
        candidates = gt_by_image_and_label.get(key, [])
        best_iou = 0.0
        best_index: int | None = None
        for index, ground_truth in enumerate(candidates):
            if index in matched_indices.get(key, set()):
                continue
            iou = bbox_iou(prediction.bbox, ground_truth.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_thr:
            matched_indices[key].add(best_index)
            true_positives.append(MatchedPrediction(prediction, candidates[best_index], best_iou))
        else:
            false_positives.append(prediction)

    false_negatives: list[GroundTruthBox] = []
    for key, candidates in gt_by_image_and_label.items():
        used = matched_indices.get(key, set())
        false_negatives.extend(
            ground_truth for index, ground_truth in enumerate(candidates) if index not in used
        )

    return true_positives, false_positives, false_negatives


def _build_confusion_matrix(
    ground_truths: Sequence[GroundTruthBox],
    predictions: Sequence[PredictionBox],
    *,
    num_classes: int,
    iou_thr: float,
    score_thr: float,
) -> list[list[int]]:
    size = num_classes + 1
    background_index = num_classes
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    filtered_predictions = sorted(
        (prediction for prediction in predictions if prediction.score >= score_thr),
        key=lambda prediction: prediction.score,
        reverse=True,
    )
    gt_by_image = _group_ground_truths_by_image(ground_truths)
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}

    for prediction in filtered_predictions:
        candidates = gt_by_image.get(prediction.image_id, [])
        best_iou = 0.0
        best_index: int | None = None
        for index, ground_truth in enumerate(candidates):
            if index in matched[prediction.image_id]:
                continue
            iou = bbox_iou(prediction.bbox, ground_truth.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_thr:
            matched[prediction.image_id].add(best_index)
            actual = candidates[best_index].label
            predicted = prediction.label
            if 0 <= actual < num_classes and 0 <= predicted < num_classes:
                matrix[actual][predicted] += 1
            else:
                matrix[actual if 0 <= actual < num_classes else background_index][background_index] += 1
        elif 0 <= prediction.label < num_classes:
            matrix[background_index][prediction.label] += 1

    for image_id, candidates in gt_by_image.items():
        used = matched.get(image_id, set())
        for index, ground_truth in enumerate(candidates):
            if index not in used and 0 <= ground_truth.label < num_classes:
                matrix[ground_truth.label][background_index] += 1

    return matrix


def _box_to_dict(box: GroundTruthBox | PredictionBox, class_names: Sequence[str]) -> dict[str, Any]:
    payload = asdict(box)
    payload["bbox"] = list(box.bbox)
    payload["class_name"] = class_names[box.label] if 0 <= box.label < len(class_names) else str(box.label)
    return payload


def compute_detection_metrics(
    ground_truths: Sequence[GroundTruthBox],
    predictions: Sequence[PredictionBox],
    *,
    class_names: Sequence[str],
    iou_thrs: Sequence[float] | None = None,
    score_thr: float = 0.1,
    small_object_area: float = 32.0 * 32.0,
) -> EvaluationResult:
    iou_thresholds = tuple(iou_thrs or tuple(0.5 + index * 0.05 for index in range(10)))
    num_classes = len(class_names)
    metrics: dict[str, float] = {}

    per_iou_aps: dict[float, list[float]] = {}
    per_class_ap_50_95: dict[int, list[float]] = {class_id: [] for class_id in range(num_classes)}
    for iou_thr in iou_thresholds:
        aps_for_threshold: list[float] = []
        for class_id in range(num_classes):
            ap = average_precision(ground_truths, predictions, class_id=class_id, iou_thr=iou_thr)
            if ap is None:
                continue
            aps_for_threshold.append(ap)
            per_class_ap_50_95[class_id].append(ap)
            if abs(iou_thr - 0.5) < 1e-9:
                class_key = _sanitize_metric_name(class_names[class_id])
                metrics[f"AP@0.5/{class_key}"] = ap
        per_iou_aps[iou_thr] = aps_for_threshold

    metrics["mAP@0.5"] = _mean(per_iou_aps.get(0.5, []))
    metrics["mAP@0.5:0.95"] = _mean(
        ap for values in per_iou_aps.values() for ap in values
    )
    for class_id, values in per_class_ap_50_95.items():
        if values:
            class_key = _sanitize_metric_name(class_names[class_id])
            metrics[f"AP@0.5:0.95/{class_key}"] = _mean(values)

    true_positives, false_positives, false_negatives = match_predictions(
        ground_truths,
        predictions,
        iou_thr=0.5,
        score_thr=score_thr,
    )
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    precision = _safe_divide(tp_count, tp_count + fp_count)
    recall = _safe_divide(tp_count, tp_count + fn_count)
    metrics["precision@0.5"] = precision
    metrics["recall@0.5"] = recall
    metrics["F1@0.5"] = _safe_divide(2.0 * precision * recall, precision + recall)
    metrics["true_positives@0.5"] = float(tp_count)
    metrics["false_positives@0.5"] = float(fp_count)
    metrics["false_negatives@0.5"] = float(fn_count)
    metrics["mean_IoU@0.5"] = _mean(match.iou for match in true_positives)
    metrics["small_objects/mean_IoU@0.5"] = _mean(
        match.iou for match in true_positives if match.ground_truth.area <= small_object_area
    )

    for class_id, class_name in enumerate(class_names):
        class_key = _sanitize_metric_name(class_name)
        class_tp = [match for match in true_positives if match.ground_truth.label == class_id]
        class_fp = [prediction for prediction in false_positives if prediction.label == class_id]
        class_fn = [ground_truth for ground_truth in false_negatives if ground_truth.label == class_id]
        class_precision = _safe_divide(len(class_tp), len(class_tp) + len(class_fp))
        class_recall = _safe_divide(len(class_tp), len(class_tp) + len(class_fn))
        metrics[f"precision@0.5/{class_key}"] = class_precision
        metrics[f"recall@0.5/{class_key}"] = class_recall
        metrics[f"F1@0.5/{class_key}"] = _safe_divide(
            2.0 * class_precision * class_recall,
            class_precision + class_recall,
        )
        metrics[f"IoU@0.5/{class_key}"] = _mean(match.iou for match in class_tp)

    confusion_matrix = _build_confusion_matrix(
        ground_truths,
        predictions,
        num_classes=num_classes,
        iou_thr=0.5,
        score_thr=score_thr,
    )
    false_positive_payload = [_box_to_dict(prediction, class_names) for prediction in false_positives]
    false_negative_payload = [_box_to_dict(ground_truth, class_names) for ground_truth in false_negatives]
    return EvaluationResult(metrics, confusion_matrix, false_positive_payload, false_negative_payload)


def write_evaluation_artifacts(
    artifact_dir: Path,
    result: EvaluationResult,
    *,
    class_names: Sequence[str],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "false_positives.json").write_text(
        json.dumps(result.false_positives, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "false_negatives.json").write_text(
        json.dumps(result.false_negatives, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    labels = [*class_names, "__background__"]
    rows = ["actual\\pred," + ",".join(labels)]
    for label, row in zip(labels, result.confusion_matrix):
        rows.append(label + "," + ",".join(str(value) for value in row))
    (artifact_dir / "confusion_matrix.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _to_python_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    elif hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except TypeError:
            pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return list(value)


def _instance_field(instances: Any, key: str) -> Any:
    if instances is None:
        return None
    if isinstance(instances, dict):
        return instances.get(key)
    if hasattr(instances, key):
        return getattr(instances, key)
    try:
        return instances[key]
    except Exception:
        return None


def _sample_field(sample: Any, key: str) -> Any:
    if isinstance(sample, dict):
        if key in sample:
            return sample[key]
        metainfo = sample.get("metainfo")
        if isinstance(metainfo, dict):
            return metainfo.get(key)
        return None
    if hasattr(sample, key):
        return getattr(sample, key)
    metainfo = getattr(sample, "metainfo", None)
    if isinstance(metainfo, dict):
        return metainfo.get(key)
    return None


def _boxes_from_instances(instances: Any, image_id: str, *, with_scores: bool) -> list[GroundTruthBox | PredictionBox]:
    bboxes = _to_python_list(_instance_field(instances, "bboxes"))
    labels = _to_python_list(_instance_field(instances, "labels"))
    scores = _to_python_list(_instance_field(instances, "scores")) if with_scores else []

    boxes: list[GroundTruthBox | PredictionBox] = []
    for index, bbox in enumerate(bboxes):
        label = int(labels[index])
        box_tuple = tuple(float(value) for value in bbox)
        if len(box_tuple) != 4:
            continue
        if with_scores:
            boxes.append(PredictionBox(image_id, label, float(scores[index]), box_tuple))
        else:
            boxes.append(GroundTruthBox(image_id, label, box_tuple))
    return boxes


try:
    from mmengine.evaluator import BaseMetric
    from mmdet.registry import METRICS
except ImportError:  # pragma: no cover - optional MMDetection runtime
    BaseMetric = object  # type: ignore[assignment,misc]
    METRICS = None


if METRICS is not None:

    @METRICS.register_module()
    class YOLOTxtMetric(BaseMetric):  # type: ignore[misc]
        """Validation metric for project YOLO TXT datasets."""

        default_prefix = "energy"

        def __init__(
            self,
            class_names: Sequence[str],
            score_thr: float = 0.1,
            iou_thrs: Sequence[float] | None = None,
            small_object_area: float = 32.0 * 32.0,
            artifact_dir: str | None = None,
            collect_device: str = "cpu",
            prefix: str | None = None,
        ) -> None:
            super().__init__(collect_device=collect_device, prefix=prefix)
            self.class_names = tuple(class_names)
            self.score_thr = score_thr
            self.iou_thrs = tuple(iou_thrs or tuple(0.5 + index * 0.05 for index in range(10)))
            self.small_object_area = small_object_area
            self.artifact_dir = Path(artifact_dir) if artifact_dir else None

        def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
            for sample in data_samples:
                image_id = (
                    _sample_field(sample, "img_id")
                    or _sample_field(sample, "img_path")
                    or f"sample-{len(self.results)}"
                )
                image_id = str(image_id)
                predictions = _boxes_from_instances(
                    _sample_field(sample, "pred_instances"),
                    image_id,
                    with_scores=True,
                )
                ground_truths = _boxes_from_instances(
                    _sample_field(sample, "gt_instances"),
                    image_id,
                    with_scores=False,
                )
                self.results.append(
                    {
                        "predictions": predictions,
                        "ground_truths": ground_truths,
                    }
                )

        def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
            predictions: list[PredictionBox] = []
            ground_truths: list[GroundTruthBox] = []
            for item in results:
                predictions.extend(item["predictions"])
                ground_truths.extend(item["ground_truths"])

            result = compute_detection_metrics(
                ground_truths,
                predictions,
                class_names=self.class_names,
                iou_thrs=self.iou_thrs,
                score_thr=self.score_thr,
                small_object_area=self.small_object_area,
            )
            if self.artifact_dir is not None:
                write_evaluation_artifacts(self.artifact_dir, result, class_names=self.class_names)
            return result.metrics
