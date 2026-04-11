from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class RunDirectories:
    run_name: str
    run_dir: Path
    tensorboard_dir: Path
    local_visualizations_dir: Path
    checkpoint_dir: Path
    inference_output_dir: Path


def build_run_directories(path_config: Any, run_name: str, inference_subdir: str) -> RunDirectories:
    run_dir = path_config.log_root / run_name
    return RunDirectories(
        run_name=run_name,
        run_dir=run_dir,
        tensorboard_dir=path_config.log_root / "tensorboard" / run_name,
        local_visualizations_dir=path_config.log_root / "local" / run_name,
        checkpoint_dir=path_config.checkpoint_root / run_name,
        inference_output_dir=path_config.output_root / inference_subdir,
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def list_image_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {input_path}")
    return image_paths


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return to_serializable(vars(value))
    return str(value)
