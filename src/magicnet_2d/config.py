from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


@dataclass(frozen=True)
class ProjectMetadata:
    name: str
    run_name: str


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    data_root: Path
    log_root: Path
    checkpoint_root: Path
    output_root: Path


@dataclass(frozen=True)
class MMDetConfig:
    config_ref: str
    checkpoint: Path | None
    disable_pretrained: bool


@dataclass(frozen=True)
class DatasetSplitConfig:
    ann_file: str
    img_prefix: str

    def ann_path(self, data_root: Path) -> Path:
        return data_root / self.ann_file

    def image_root(self, data_root: Path) -> Path:
        return data_root / self.img_prefix


@dataclass(frozen=True)
class DatasetConfig:
    dataset_type: str
    classes: tuple[str, ...]
    train: DatasetSplitConfig
    val: DatasetSplitConfig
    test: DatasetSplitConfig


@dataclass(frozen=True)
class TrainSettings:
    seed: int
    validate: bool
    amp: bool
    max_epochs: int | None
    num_workers: int | None
    checkpoint_interval: int
    max_keep_checkpoints: int
    logger_interval: int
    resume: Path | None


@dataclass(frozen=True)
class InferenceSettings:
    device: str
    score_threshold: float
    batch_size: int
    output_subdir: str


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    project: ProjectMetadata
    paths: PathConfig
    mmdet: MMDetConfig
    dataset: DatasetConfig
    train: TrainSettings
    inference: InferenceSettings


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table in the TOML config.")
    return value


def _as_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _as_optional_string(value: Any, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _as_string(value, label)


def _as_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array of strings.")
    converted: list[str] = []
    for item in value:
        converted.append(_as_string(item, label))
    return tuple(converted)


def _as_split(value: Any, label: str) -> DatasetSplitConfig:
    mapping = _as_mapping(value, label)
    return DatasetSplitConfig(
        ann_file=_as_string(mapping.get("ann_file"), f"{label}.ann_file"),
        img_prefix=_as_string(mapping.get("img_prefix"), f"{label}.img_prefix"),
    )


def load_project_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project_root = _find_project_root(config_path.parent)

    project_raw = _as_mapping(raw.get("project"), "project")
    paths_raw = _as_mapping(raw.get("paths"), "paths")
    mmdet_raw = _as_mapping(raw.get("mmdet"), "mmdet")
    dataset_raw = _as_mapping(raw.get("dataset"), "dataset")
    train_raw = _as_mapping(raw.get("train"), "train")
    inference_raw = _as_mapping(raw.get("inference"), "inference")

    project = ProjectMetadata(
        name=_as_string(project_raw.get("name"), "project.name"),
        run_name=_as_string(project_raw.get("run_name"), "project.run_name"),
    )
    paths = PathConfig(
        project_root=project_root,
        data_root=_resolve_project_path(project_root, _as_string(paths_raw.get("data_root"), "paths.data_root")),
        log_root=_resolve_project_path(project_root, _as_string(paths_raw.get("log_root"), "paths.log_root")),
        checkpoint_root=_resolve_project_path(
            project_root, _as_string(paths_raw.get("checkpoint_root"), "paths.checkpoint_root")
        ),
        output_root=_resolve_project_path(project_root, _as_string(paths_raw.get("output_root"), "paths.output_root")),
    )
    mmdet = MMDetConfig(
        config_ref=_as_string(mmdet_raw.get("config"), "mmdet.config"),
        checkpoint=(
            _resolve_project_path(project_root, checkpoint)
            if (checkpoint := _as_optional_string(mmdet_raw.get("checkpoint"), "mmdet.checkpoint"))
            else None
        ),
        disable_pretrained=bool(mmdet_raw.get("disable_pretrained", False)),
    )
    dataset = DatasetConfig(
        dataset_type=_as_string(dataset_raw.get("type"), "dataset.type"),
        classes=_as_string_tuple(dataset_raw.get("classes"), "dataset.classes"),
        train=_as_split(dataset_raw.get("train"), "dataset.train"),
        val=_as_split(dataset_raw.get("val"), "dataset.val"),
        test=_as_split(dataset_raw.get("test"), "dataset.test"),
    )
    train = TrainSettings(
        seed=int(train_raw.get("seed", 42)),
        validate=bool(train_raw.get("validate", True)),
        amp=bool(train_raw.get("amp", False)),
        max_epochs=int(train_raw["max_epochs"]) if train_raw.get("max_epochs") is not None else None,
        num_workers=int(train_raw["num_workers"]) if train_raw.get("num_workers") is not None else None,
        checkpoint_interval=int(train_raw.get("checkpoint_interval", 1)),
        max_keep_checkpoints=int(train_raw.get("max_keep_checkpoints", 3)),
        logger_interval=int(train_raw.get("logger_interval", 50)),
        resume=(
            _resolve_project_path(project_root, resume)
            if (resume := _as_optional_string(train_raw.get("resume"), "train.resume"))
            else None
        ),
    )
    inference = InferenceSettings(
        device=_as_string(inference_raw.get("device", "auto"), "inference.device"),
        score_threshold=float(inference_raw.get("score_threshold", 0.3)),
        batch_size=int(inference_raw.get("batch_size", 1)),
        output_subdir=_as_string(inference_raw.get("output_subdir", "inference"), "inference.output_subdir"),
    )

    if not 0.0 <= inference.score_threshold <= 1.0:
        raise ValueError("inference.score_threshold must be between 0 and 1.")

    return ProjectConfig(
        config_path=config_path,
        project=project,
        paths=paths,
        mmdet=mmdet,
        dataset=dataset,
        train=train,
        inference=inference,
    )


def override_config(
    config: ProjectConfig,
    *,
    run_name: str | None = None,
    mmdet_config_ref: str | None = None,
    checkpoint: Path | None = None,
    inference_device: str | None = None,
    score_threshold: float | None = None,
) -> ProjectConfig:
    project = replace(config.project, run_name=run_name or config.project.run_name)
    mmdet = replace(
        config.mmdet,
        config_ref=mmdet_config_ref or config.mmdet.config_ref,
        checkpoint=checkpoint if checkpoint is not None else config.mmdet.checkpoint,
    )
    inference = replace(
        config.inference,
        device=inference_device or config.inference.device,
        score_threshold=score_threshold if score_threshold is not None else config.inference.score_threshold,
    )
    return replace(config, project=project, mmdet=mmdet, inference=inference)


def missing_training_paths(config: ProjectConfig) -> list[Path]:
    paths = [
        config.dataset.train.ann_path(config.paths.data_root),
        config.dataset.train.image_root(config.paths.data_root),
    ]
    if config.train.validate:
        paths.extend(
            [
                config.dataset.val.ann_path(config.paths.data_root),
                config.dataset.val.image_root(config.paths.data_root),
            ]
        )
    return [path for path in paths if not path.exists()]
