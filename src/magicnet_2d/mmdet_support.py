from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .utils import RunDirectories, build_run_directories


def _missing_runtime_message() -> str:
    return (
        "MMDetection runtime is not installed. "
        "Run `python scripts/check_env.py check` to inspect the environment and "
        "`python scripts/install_system_deps.py install --execute` or `uv sync --extra cpu --extra dev` "
        "to install a compatible runtime."
    )


def _require_mmdet_training_runtime() -> tuple[Any, Any]:
    try:
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
    except ImportError as exc:  # pragma: no cover - exercised only with real runtime
        raise RuntimeError(_missing_runtime_message()) from exc
    register_all_modules(init_default_scope=False)
    return Config, register_all_modules


def resolve_mmdet_config(config_ref: str, project_root: Path) -> Path:
    if config_ref.startswith("mmdet::"):
        suffix = config_ref.split("::", 1)[1]
        try:
            import mmdet
        except ImportError as exc:  # pragma: no cover - exercised only with real runtime
            raise RuntimeError(_missing_runtime_message()) from exc

        package_root = Path(mmdet.__file__).resolve().parent
        candidates = [
            package_root / ".mim" / "configs" / suffix,
            package_root.parent / ".mim" / "configs" / suffix,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"MMDetection config was not found: {config_ref}")

    path = Path(config_ref)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return path


def _walk(node: Any) -> list[Any]:
    if isinstance(node, dict):
        nested = [node]
        for value in node.values():
            nested.extend(_walk(value))
        return nested
    if isinstance(node, list):
        nested: list[Any] = []
        for value in node:
            nested.extend(_walk(value))
        return nested
    return []


def _patch_dataset_nodes(dataset_node: Any, config: ProjectConfig, split_name: str) -> None:
    split = getattr(config.dataset, split_name)
    classes = list(config.dataset.classes)

    for node in _walk(dataset_node):
        if not isinstance(node, dict):
            continue
        if "datasets" in node and isinstance(node["datasets"], list):
            continue
        if "dataset" in node and isinstance(node["dataset"], dict):
            continue
        if {"ann_file", "data_prefix"} & set(node.keys()) or node.get("type") == config.dataset.dataset_type:
            node["type"] = config.dataset.dataset_type
            node["data_root"] = str(config.paths.data_root)
            node["ann_file"] = split.ann_file
            node["data_prefix"] = {"img": split.img_prefix}
            node["metainfo"] = {"classes": classes}


def _patch_evaluator(evaluator_node: Any, data_root: Path, ann_file: str) -> None:
    for node in _walk(evaluator_node):
        if isinstance(node, dict) and "ann_file" in node:
            node["ann_file"] = str(data_root / ann_file)


def _patch_num_classes(model_node: Any, num_classes: int) -> None:
    for node in _walk(model_node):
        if isinstance(node, dict) and "num_classes" in node:
            node["num_classes"] = num_classes


def _configure_hooks(cfg: Any, config: ProjectConfig, run_dirs: RunDirectories) -> None:
    default_hooks = dict(cfg.get("default_hooks", {}))

    logger_hook = dict(default_hooks.get("logger", {}))
    logger_hook.setdefault("type", "LoggerHook")
    logger_hook["interval"] = config.train.logger_interval
    default_hooks["logger"] = logger_hook

    checkpoint_hook = dict(default_hooks.get("checkpoint", {}))
    checkpoint_hook.setdefault("type", "CheckpointHook")
    checkpoint_hook["interval"] = config.train.checkpoint_interval
    checkpoint_hook["max_keep_ckpts"] = config.train.max_keep_checkpoints
    checkpoint_hook["out_dir"] = str(run_dirs.checkpoint_dir)
    default_hooks["checkpoint"] = checkpoint_hook

    default_hooks.setdefault("timer", {"type": "IterTimerHook"})
    default_hooks.setdefault("param_scheduler", {"type": "ParamSchedulerHook"})
    default_hooks.setdefault("sampler_seed", {"type": "DistSamplerSeedHook"})
    default_hooks.setdefault("visualization", {"type": "DetVisualizationHook"})
    cfg.default_hooks = default_hooks


def _configure_visualizer(cfg: Any, run_dirs: RunDirectories) -> None:
    cfg.visualizer = {
        "type": "DetLocalVisualizer",
        "vis_backends": [
            {"type": "LocalVisBackend", "save_dir": str(run_dirs.local_visualizations_dir)},
            {"type": "TensorboardVisBackend", "save_dir": str(run_dirs.tensorboard_dir)},
        ],
        "name": "visualizer",
    }


def build_runtime_config(
    project_config: ProjectConfig,
    *,
    mmdet_config_override: str | None = None,
    checkpoint_override: Path | None = None,
) -> tuple[Any, RunDirectories, Path]:
    Config, _ = _require_mmdet_training_runtime()
    run_dirs = build_run_directories(
        project_config.paths,
        project_config.project.run_name,
        project_config.inference.output_subdir,
    )
    config_path = resolve_mmdet_config(
        mmdet_config_override or project_config.mmdet.config_ref,
        project_config.paths.project_root,
    )
    cfg = Config.fromfile(str(config_path))

    cfg.default_scope = "mmdet"
    cfg.work_dir = str(run_dirs.run_dir)
    cfg.log_level = cfg.get("log_level", "INFO")
    cfg.randomness = {"seed": project_config.train.seed, "deterministic": False}

    if project_config.train.max_epochs is not None:
        train_cfg = dict(cfg.get("train_cfg", {}))
        train_cfg["max_epochs"] = project_config.train.max_epochs
        cfg.train_cfg = train_cfg

    if not project_config.train.validate:
        cfg.val_cfg = None
        cfg.val_evaluator = None

    if project_config.train.amp and cfg.get("optim_wrapper"):
        optim_wrapper = dict(cfg.optim_wrapper)
        optim_wrapper["type"] = "AmpOptimWrapper"
        optim_wrapper.setdefault("loss_scale", "dynamic")
        cfg.optim_wrapper = optim_wrapper

    _patch_dataset_nodes(cfg.get("train_dataloader"), project_config, "train")
    _patch_dataset_nodes(cfg.get("val_dataloader"), project_config, "val")
    _patch_dataset_nodes(cfg.get("test_dataloader"), project_config, "test")
    _patch_evaluator(cfg.get("val_evaluator"), project_config.paths.data_root, project_config.dataset.val.ann_file)
    _patch_evaluator(cfg.get("test_evaluator"), project_config.paths.data_root, project_config.dataset.test.ann_file)
    _patch_num_classes(cfg.get("model"), len(project_config.dataset.classes))
    _configure_hooks(cfg, project_config, run_dirs)
    _configure_visualizer(cfg, run_dirs)

    checkpoint = checkpoint_override or project_config.mmdet.checkpoint
    if project_config.train.resume:
        cfg.resume = True
        cfg.load_from = str(project_config.train.resume)
    elif checkpoint is not None:
        cfg.load_from = str(checkpoint)
        cfg.resume = False
    else:
        cfg.resume = False

    return cfg, run_dirs, config_path
