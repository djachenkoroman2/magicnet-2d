from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

COCO80_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass(frozen=True)
class YOLORunPaths:
    project_root: Path
    log_project_dir: Path
    run_dir: Path
    checkpoint_dir: Path
    output_project_dir: Path


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def resolve_candidate_path(base_dir: Path, value: str, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    preferred = (base_dir / path).resolve()
    if preferred.exists():
        return preferred
    return (project_root / path).resolve()


def maybe_resolve_model(value: str, project_root: Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate.resolve())
    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return str(project_candidate)
    return value


def build_yolo_run_paths(project_root: Path, run_name: str) -> YOLORunPaths:
    log_project_dir = project_root / "log" / "yolo"
    output_project_dir = project_root / "outputs" / "yolo"
    return YOLORunPaths(
        project_root=project_root,
        log_project_dir=log_project_dir,
        run_dir=log_project_dir / run_name,
        checkpoint_dir=project_root / "checkpoints" / "yolo" / run_name,
        output_project_dir=output_project_dir,
    )


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def prepare_ultralytics_environment(project_root: Path) -> Path:
    config_root = project_root / ".ultralytics"
    matplotlib_root = project_root / ".matplotlib"
    config_root.mkdir(parents=True, exist_ok=True)
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root))
    return config_root


def materialize_dataset_yaml(source_yaml: Path, destination_yaml: Path, dataset_root: Path) -> Path:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("PyYAML is required to prepare the YOLO dataset config.") from exc

    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source_yaml}")

    payload = dict(payload)
    payload["path"] = str(dataset_root.resolve())
    destination_yaml.parent.mkdir(parents=True, exist_ok=True)
    destination_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination_yaml


def copy_weight_artifacts(source_weights_dir: Path, checkpoint_dir: Path) -> list[Path]:
    copied: list[Path] = []
    if not source_weights_dir.exists():
        return copied

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("best.pt", "last.pt"):
        source = source_weights_dir / filename
        if not source.exists():
            continue
        destination = checkpoint_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
