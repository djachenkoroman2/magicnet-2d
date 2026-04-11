from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .yolo_support import ensure_directories, find_project_root


def _require_inference_runtime() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Ultralytics is not installed. Run `uv sync --extra yolo --extra dev`.") from exc
    return YOLO


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to a YOLO checkpoint, for example checkpoints/yolo/coco8_smoke/best.pt.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image or directory for inference.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the annotated predictions will be saved.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example `cpu` or `0`.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name used when --output-dir is not provided.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {weights_path}")

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Inference source does not exist: {source_path}")

    project_root = find_project_root(Path.cwd())
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        run_name = args.name or f"{source_path.stem}_pred"
        output_dir = project_root / "outputs" / "yolo" / run_name

    ensure_directories([output_dir.parent])

    YOLO = _require_inference_runtime()
    model = YOLO(str(weights_path))
    model.predict(
        source=str(source_path),
        conf=float(args.conf),
        device=args.device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
    )

    print(f"Inference source: {source_path}")
    print(f"Checkpoint: {weights_path}")
    print(f"Results directory: {output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet-yolo-infer",
        description="Run inference with a trained Ultralytics YOLO checkpoint.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    return run_from_args(args)
