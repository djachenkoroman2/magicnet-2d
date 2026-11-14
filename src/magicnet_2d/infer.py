from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import time
from typing import Sequence

from .config import load_project_config, override_config
from .mmdet_support import MissingMMDetectionRuntimeError, build_runtime_config
from .utils import choose_device, ensure_directories, list_image_inputs, to_serializable


def _resolve_checkpoint(config_checkpoint: Path | None, cli_checkpoint: str | None) -> Path:
    if cli_checkpoint:
        checkpoint = Path(cli_checkpoint).resolve()
    elif config_checkpoint:
        checkpoint = config_checkpoint.resolve()
    else:
        raise ValueError("A checkpoint is required for inference. Pass --checkpoint or set mmdet.checkpoint in the config.")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def _summarize_visualization(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return [_summarize_visualization(item) for item in value]

    shape = getattr(value, "shape", None)
    if shape is not None:
        dtype = getattr(value, "dtype", None)
        return {
            "type": type(value).__name__,
            "shape": [int(dimension) for dimension in shape],
            "dtype": str(dtype) if dtype is not None else None,
        }

    return to_serializable(value)


def _summarize_inferencer_results(results: object) -> object:
    if not isinstance(results, dict):
        return to_serializable(results)

    summary: dict[str, object] = {}
    for key, value in results.items():
        if key == "visualization":
            summary[key] = _summarize_visualization(value)
        else:
            summary[key] = to_serializable(value)
    return summary


def _batched(items: Sequence[Path], batch_size: int) -> list[list[Path]]:
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


def _merge_inferencer_summaries(summaries: Sequence[object]) -> object:
    merged: dict[str, list[object]] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            merged.setdefault("items", []).append(summary)
            continue

        for key, value in summary.items():
            if isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            else:
                merged.setdefault(key, []).append(value)
    return merged


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/project/default.toml",
        help="Path to the project TOML config.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory with images for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained MMDetection checkpoint.",
    )
    parser.add_argument(
        "--mmdet-config",
        default=None,
        help="Override the MMDetection config reference.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where visualizations and predictions will be saved.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help="Override the score threshold used by DetInferencer.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example `cpu`, `cuda:0` or `auto`.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    project_config = load_project_config(args.config)
    project_config = override_config(
        project_config,
        mmdet_config_ref=args.mmdet_config,
        inference_device=args.device,
        score_threshold=args.score_thr,
    )

    checkpoint_path = _resolve_checkpoint(project_config.mmdet.checkpoint, args.checkpoint)
    input_path = Path(args.input).resolve()
    input_items = list_image_inputs(input_path)

    try:
        cfg, run_dirs, _ = build_runtime_config(project_config, checkpoint_override=checkpoint_path)
    except MissingMMDetectionRuntimeError as exc:
        print(exc)
        return 1
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dirs.inference_output_dir.resolve()
    ensure_directories([output_dir])

    resolved_config_path = output_dir / "resolved_inference_config.py"
    cfg.dump(str(resolved_config_path))

    try:
        from mmdet.apis import DetInferencer
    except ImportError as exc:  # pragma: no cover - requires real runtime
        raise RuntimeError(
            "MMDetection inference runtime is not installed. Run `python scripts/check_env.py check` first."
        ) from exc

    inferencer = DetInferencer(
        model=str(resolved_config_path),
        weights=str(checkpoint_path),
        device=choose_device(project_config.inference.device),
    )

    call_signature = inspect.signature(inferencer.__call__)
    call_kwargs: dict[str, object] = {}
    if "pred_score_thr" in call_signature.parameters:
        call_kwargs["pred_score_thr"] = project_config.inference.score_threshold
    if "out_dir" in call_signature.parameters:
        call_kwargs["out_dir"] = str(output_dir)
    if "batch_size" in call_signature.parameters:
        call_kwargs["batch_size"] = max(1, project_config.inference.batch_size)
    if "no_save_pred" in call_signature.parameters:
        call_kwargs["no_save_pred"] = False
    if "no_save_vis" in call_signature.parameters:
        call_kwargs["no_save_vis"] = False

    result_summaries: list[object] = []
    input_batches = _batched(input_items, max(1, project_config.inference.batch_size))
    started_at = time.perf_counter()
    for batch_index, input_batch in enumerate(input_batches, start=1):
        print(f"Running inference batch {batch_index}/{len(input_batches)} ({len(input_batch)} item(s)).")
        results = inferencer([str(item) for item in input_batch], **call_kwargs)
        result_summaries.append(_summarize_inferencer_results(results))
    elapsed_seconds = time.perf_counter() - started_at
    image_count = len(input_items)
    speed_metrics = {
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": image_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "latency_seconds_per_image": elapsed_seconds / image_count if image_count else 0.0,
        "batch_size": max(1, project_config.inference.batch_size),
        "device": choose_device(project_config.inference.device),
    }

    summary_path = output_dir / "predictions.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_config": str(resolved_config_path),
                "checkpoint": str(checkpoint_path),
                "device": choose_device(project_config.inference.device),
                "inputs": [str(item) for item in input_items],
                "speed": speed_metrics,
                "results": _merge_inferencer_summaries(result_summaries),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Inference finished for {len(input_items)} item(s).")
    print(f"Resolved config: {resolved_config_path}")
    print(f"Predictions summary: {summary_path}")
    print(f"Artifacts directory: {output_dir}")
    print(
        "Inference speed: "
        f"{speed_metrics['images_per_second']:.3f} images/s, "
        f"{speed_metrics['latency_seconds_per_image']:.3f} s/image."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet-infer",
        description="Run inference for a single image or a directory of images using MMDetection.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    return run_from_args(args)
