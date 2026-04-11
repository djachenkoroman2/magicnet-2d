from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Sequence

from .config import load_project_config, override_config
from .mmdet_support import build_runtime_config
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

    cfg, run_dirs, _ = build_runtime_config(project_config, checkpoint_override=checkpoint_path)
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
        call_kwargs["batch_size"] = project_config.inference.batch_size
    if "no_save_pred" in call_signature.parameters:
        call_kwargs["no_save_pred"] = False
    if "no_save_vis" in call_signature.parameters:
        call_kwargs["no_save_vis"] = False

    results = inferencer([str(item) for item in input_items], **call_kwargs)
    summary_path = output_dir / "predictions.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_config": str(resolved_config_path),
                "checkpoint": str(checkpoint_path),
                "device": choose_device(project_config.inference.device),
                "inputs": [str(item) for item in input_items],
                "results": to_serializable(results),
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
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet-infer",
        description="Run inference for a single image or a directory of images using MMDetection.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    return run_from_args(args)
