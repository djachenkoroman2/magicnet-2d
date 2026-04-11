from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .yolo_support import (
    build_yolo_run_paths,
    copy_weight_artifacts,
    ensure_directories,
    find_project_root,
    maybe_resolve_model,
    resolve_candidate_path,
    write_summary,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PyYAML is not installed. Run `uv sync --extra yolo --extra dev`.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YOLO config must contain a mapping: {path}")
    return payload


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    if args.model:
        updated["model"] = args.model
    if args.run_name:
        updated["name"] = args.run_name
    if args.epochs is not None:
        updated["epochs"] = args.epochs
    if args.device:
        updated["device"] = args.device
    if args.batch is not None:
        updated["batch"] = args.batch
    if args.imgsz is not None:
        updated["imgsz"] = args.imgsz
    return updated


def _require_training_runtime() -> tuple[Any, Any]:
    try:
        from ultralytics import YOLO, settings
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Ultralytics is not installed. Run `uv sync --extra yolo --extra dev`.") from exc
    return YOLO, settings


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/yolo/coco8_smoke.yaml",
        help="Path to the YOLO training YAML config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the configured run name.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured YOLO model or local weights file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example `cpu`, `0`, or `0,1`.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured number of epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override the configured image size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the resolved training plan without launching YOLO.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"YOLO config does not exist: {config_path}")

    project_root = find_project_root(config_path.parent)
    train_config = _apply_overrides(_load_yaml(config_path), args)

    if "data" not in train_config or "model" not in train_config or "name" not in train_config:
        raise ValueError("YOLO config must contain at least `data`, `model`, and `name`.")

    dataset_config_path = resolve_candidate_path(config_path.parent, str(train_config["data"]), project_root)
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_config_path}")

    run_name = str(train_config["name"])
    run_paths = build_yolo_run_paths(project_root, run_name)
    ensure_directories([run_paths.log_project_dir, run_paths.checkpoint_dir, run_paths.output_project_dir])

    resolved_model = maybe_resolve_model(str(train_config["model"]), project_root)
    dry_run_lines = [
        f"YOLO config: {config_path}",
        f"Dataset YAML: {dataset_config_path}",
        f"Model: {resolved_model}",
        f"Run dir: {run_paths.run_dir}",
        f"Checkpoint dir: {run_paths.checkpoint_dir}",
        f"TensorBoard dir: {run_paths.run_dir}",
    ]

    if args.dry_run:
        print("\n".join(dry_run_lines))
        return 0

    YOLO, settings = _require_training_runtime()
    try:
        settings.update({"tensorboard": True})
    except Exception:
        pass

    model = YOLO(resolved_model)
    train_kwargs = dict(train_config)
    train_kwargs["data"] = str(dataset_config_path)
    train_kwargs["model"] = resolved_model
    train_kwargs["project"] = str(run_paths.log_project_dir)
    train_kwargs["name"] = run_name
    train_kwargs.setdefault("exist_ok", True)
    train_kwargs.setdefault("plots", True)
    train_kwargs.setdefault("save", True)
    train_kwargs.setdefault("workers", 0)
    train_kwargs.setdefault("seed", 42)
    train_kwargs.pop("notes", None)

    model.train(**train_kwargs)

    copied_weights = copy_weight_artifacts(run_paths.run_dir / "weights", run_paths.checkpoint_dir)
    summary = {
        "train_config": str(config_path),
        "dataset_yaml": str(dataset_config_path),
        "model": resolved_model,
        "run_dir": str(run_paths.run_dir),
        "tensorboard_dir": str(run_paths.run_dir),
        "checkpoint_dir": str(run_paths.checkpoint_dir),
        "copied_weights": [str(path) for path in copied_weights],
    }
    write_summary(run_paths.run_dir / "run_summary.json", summary)

    print(f"Training finished: {run_paths.run_dir}")
    print(f"TensorBoard logs: {run_paths.run_dir}")
    print(f"Copied checkpoints: {run_paths.checkpoint_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet-yolo-train",
        description="Train an Ultralytics YOLO detector on the local smoke-test dataset.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    return run_from_args(args)
