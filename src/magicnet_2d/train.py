from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_project_config, missing_training_paths, override_config
from .mmdet_support import build_runtime_config
from .utils import ensure_directories


def _format_missing_paths(paths: list[Path]) -> str:
    lines = ["Training dataset paths are missing:"]
    lines.extend(f"  - {path}" for path in paths)
    return "\n".join(lines)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/project/default.toml",
        help="Path to the project TOML config.",
    )
    parser.add_argument(
        "--mmdet-config",
        default=None,
        help="Override the MMDetection config reference.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint to load before training.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the configured run name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the resolved MMDetection config and validate paths without starting training.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    project_config = load_project_config(args.config)
    project_config = override_config(
        project_config,
        run_name=args.run_name,
        mmdet_config_ref=args.mmdet_config,
        checkpoint=checkpoint,
    )

    missing_paths = missing_training_paths(project_config)
    if missing_paths and not args.dry_run:
        raise FileNotFoundError(_format_missing_paths(missing_paths))

    cfg, run_dirs, source_config_path = build_runtime_config(project_config)
    ensure_directories(
        [
            run_dirs.run_dir,
            run_dirs.tensorboard_dir,
            run_dirs.local_visualizations_dir,
            run_dirs.checkpoint_dir,
        ]
    )

    resolved_config_path = run_dirs.run_dir / "resolved_config.py"
    cfg.dump(str(resolved_config_path))

    if missing_paths and args.dry_run:
        print(_format_missing_paths(missing_paths))
        print()

    if args.dry_run:
        print(f"Resolved MMDetection config source: {source_config_path}")
        print(f"Resolved runtime config saved to: {resolved_config_path}")
        print(f"TensorBoard logs will be written to: {run_dirs.tensorboard_dir}")
        print(f"Checkpoints will be written to: {run_dirs.checkpoint_dir}")
        return 0

    from mmengine.runner import Runner  # pragma: no cover - requires real runtime

    runner = Runner.from_cfg(cfg)
    runner.train()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet-train",
        description="Train a 2D detector using the project TOML config and an MMDetection base config.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    return run_from_args(args)
