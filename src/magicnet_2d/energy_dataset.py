from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import shutil
from typing import Sequence, TypeVar
from urllib.parse import unquote


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
GEO_SUFFIXES = (".jgw", ".prj")
UUID_LABEL_RE = re.compile(r"^[0-9a-fA-F]{8}-(?P<image_stem>.+)$")
MANAGED_OUTPUT_NAMES = ("images", "labels", "geo", "README.md", "metadata.json", "classes.txt")


@dataclass(frozen=True)
class EnergySourceItem:
    project_name: str
    image_path: Path
    label_path: Path | None
    geo_paths: tuple[Path, ...]

    @property
    def relative_image_path(self) -> Path:
        return Path(self.image_path.name)

    @property
    def relative_label_path(self) -> Path:
        return self.relative_image_path.with_suffix(".txt")


@dataclass(frozen=True)
class ResolvedLabelConflict:
    image_stem: str
    kept_label_path: Path
    rejected_label_paths: tuple[Path, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "image_stem": self.image_stem,
            "kept_label_path": str(self.kept_label_path),
            "rejected_label_paths": [str(path) for path in self.rejected_label_paths],
        }


@dataclass(frozen=True)
class DatasetClassConfig:
    classes: tuple[str, ...]
    class_id_map: tuple[int, ...] | None


@dataclass(frozen=True)
class EnergyDatasetPlan:
    source_root: Path
    output_root: Path
    categories_by_id: dict[int, str]
    class_config: DatasetClassConfig
    class_counts: Counter[int]
    unmatched_label_paths: tuple[Path, ...]
    splits: dict[str, tuple[EnergySourceItem, ...]]
    merge_source_roots: tuple[Path, ...] = ()
    preserved_unmatched_labels: tuple[str, ...] = ()
    preserved_missing_label_images: tuple[str, ...] = ()
    preserved_resolved_label_conflicts: tuple[dict[str, object], ...] = ()
    resolved_label_conflicts: tuple[ResolvedLabelConflict, ...] = ()
    pruned_unlabeled: dict[str, object] | None = None
    added_splits: dict[str, tuple[EnergySourceItem, ...]] | None = None

    @property
    def total_images(self) -> int:
        return sum(len(items) for items in self.splits.values())

    @property
    def total_labels(self) -> int:
        return sum(1 for item in self.items if item.label_path is not None)

    @property
    def missing_labels(self) -> int:
        return sum(1 for item in self.items if item.label_path is None)

    @property
    def missing_label_items(self) -> tuple[EnergySourceItem, ...]:
        return tuple(item for item in self.items if item.label_path is None)

    @property
    def empty_labels(self) -> int:
        return sum(1 for item in self.items if item.label_path is not None and item.label_path.stat().st_size == 0)

    @property
    def geo_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for item in self.items:
            for geo_path in item.geo_paths:
                counts[geo_path.suffix.lower()] += 1
        return counts

    @property
    def items(self) -> tuple[EnergySourceItem, ...]:
        return tuple(item for split_items in self.splits.values() for item in split_items)

    @property
    def all_unmatched_labels(self) -> tuple[str, ...]:
        return self.preserved_unmatched_labels + tuple(str(path) for path in self.unmatched_label_paths)

    @property
    def all_missing_label_images(self) -> tuple[str, ...]:
        return self.preserved_missing_label_images + tuple(
            str(item.image_path) for item in self.missing_label_items
        )

    @property
    def all_resolved_label_conflicts(self) -> tuple[dict[str, object], ...]:
        return self.preserved_resolved_label_conflicts + tuple(
            conflict.to_metadata() for conflict in self.resolved_label_conflicts
        )


def normalize_label_stem(label_stem: str, image_stems: set[str]) -> str | None:
    candidates = [label_stem]
    decoded_stem = unquote(label_stem)
    if decoded_stem != label_stem:
        candidates.append(decoded_stem)

    for candidate in candidates:
        if candidate in image_stems:
            return candidate
        match = UUID_LABEL_RE.match(candidate)
        if match:
            image_stem = match.group("image_stem")
            if image_stem in image_stems:
                return image_stem

    suffix_matches = sorted(
        stem for stem in image_stems if label_stem.endswith(f"-{stem}") or decoded_stem.endswith(f"-{stem}")
    )
    if not suffix_matches:
        return None
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise ValueError(
        f"Label stem `{label_stem}` matches multiple image stems: {', '.join(suffix_matches)}."
    )


def read_project_categories(project_dir: Path) -> dict[int, str]:
    notes_path = project_dir / "notes.json"
    if notes_path.exists():
        raw = json.loads(notes_path.read_text(encoding="utf-8"))
        categories = raw.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"`categories` must be a non-empty list in {notes_path}.")

        parsed: dict[int, str] = {}
        for category in categories:
            if not isinstance(category, dict):
                raise ValueError(f"Invalid category entry in {notes_path}: {category!r}.")
            category_id = category.get("id")
            category_name = category.get("name")
            if not isinstance(category_id, int) or not isinstance(category_name, str) or not category_name.strip():
                raise ValueError(f"Invalid category entry in {notes_path}: {category!r}.")
            if category_id in parsed:
                raise ValueError(f"Duplicate category id {category_id} in {notes_path}.")
            parsed[category_id] = category_name.strip()
        return parsed

    classes_path = project_dir / "classes.txt"
    if classes_path.exists():
        names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not names:
            raise ValueError(f"No class names found in {classes_path}.")
        return {index: name for index, name in enumerate(names)}

    raise FileNotFoundError(f"Neither notes.json nor classes.txt was found in {project_dir}.")


def read_source_categories(source_root: Path) -> dict[int, str]:
    project_dirs = _project_dirs(source_root)
    categories_by_id: dict[int, str] | None = None
    for project_dir in project_dirs:
        project_categories = read_project_categories(project_dir)
        if categories_by_id is None:
            categories_by_id = project_categories
        elif project_categories != categories_by_id:
            raise ValueError(f"Class metadata in {project_dir} differs from previous exports.")
    if categories_by_id is None:
        raise FileNotFoundError(f"No export directories found in {source_root}.")
    return categories_by_id


def build_class_config(categories_by_id: dict[int, str], used_class_ids: set[int]) -> DatasetClassConfig:
    if not categories_by_id:
        ids = sorted(used_class_ids)
        return DatasetClassConfig(
            classes=tuple(f"source_id_{class_id}" for class_id in ids),
            class_id_map=tuple(ids) if ids != list(range(len(ids))) else None,
        )

    known_ids = sorted(categories_by_id)
    unknown_ids = sorted(used_class_ids - set(known_ids))
    if unknown_ids:
        unknown = ", ".join(str(class_id) for class_id in unknown_ids)
        raise ValueError(f"Annotation class ids are not present in notes.json/classes.txt: {unknown}.")

    if known_ids == list(range(len(known_ids))):
        return DatasetClassConfig(
            classes=tuple(categories_by_id[class_id] for class_id in known_ids),
            class_id_map=None,
        )

    selected_ids = sorted(used_class_ids or set(known_ids))
    return DatasetClassConfig(
        classes=tuple(categories_by_id[class_id] for class_id in selected_ids),
        class_id_map=tuple(selected_ids) if selected_ids != list(range(len(selected_ids))) else None,
    )


def discover_energy_items(
    source_root: Path,
) -> tuple[
    tuple[EnergySourceItem, ...],
    dict[int, str],
    Counter[int],
    tuple[Path, ...],
    tuple[ResolvedLabelConflict, ...],
]:
    source_root = source_root.resolve()
    categories_by_id = read_source_categories(source_root)
    items: list[EnergySourceItem] = []
    class_counts: Counter[int] = Counter()
    unmatched_label_paths: list[Path] = []
    resolved_conflicts: list[ResolvedLabelConflict] = []

    for project_dir in _project_dirs(source_root):
        images_dir = project_dir / "images"
        labels_dir = project_dir / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {images_dir}.")

        image_paths = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}.")

        images_by_stem: dict[str, Path] = {}
        for image_path in image_paths:
            previous = images_by_stem.setdefault(image_path.stem, image_path)
            if previous != image_path:
                raise ValueError(
                    f"Duplicate image stem `{image_path.stem}` in {images_dir}: {previous} and {image_path}."
                )

        labels_by_stem: dict[str, Path] = {}
        if labels_dir.exists():
            candidates_by_stem: dict[str, list[Path]] = {}
            for label_path in sorted(labels_dir.glob("*.txt")):
                image_stem = normalize_label_stem(label_path.stem, set(images_by_stem))
                if image_stem is None:
                    unmatched_label_paths.append(label_path)
                    continue
                candidates_by_stem.setdefault(image_stem, []).append(label_path)

            for image_stem, candidates in sorted(candidates_by_stem.items()):
                if len(candidates) == 1:
                    chosen_label = candidates[0]
                else:
                    chosen_label = max(candidates, key=lambda path: (_label_annotation_count(path), path.name))
                    resolved_conflicts.append(
                        ResolvedLabelConflict(
                            image_stem=image_stem,
                            kept_label_path=chosen_label,
                            rejected_label_paths=tuple(path for path in candidates if path != chosen_label),
                        )
                    )
                labels_by_stem[image_stem] = chosen_label
                class_counts.update(read_label_class_ids(chosen_label))

        for image_path in image_paths:
            geo_paths = tuple(
                geo_path for suffix in GEO_SUFFIXES if (geo_path := image_path.with_suffix(suffix)).exists()
            )
            items.append(
                EnergySourceItem(
                    project_name=project_dir.name,
                    image_path=image_path,
                    label_path=labels_by_stem.get(image_path.stem),
                    geo_paths=geo_paths,
                )
            )

    _validate_flattened_names(items)
    return (
        tuple(sorted(items, key=lambda item: item.relative_image_path.as_posix())),
        categories_by_id,
        class_counts,
        tuple(sorted(unmatched_label_paths)),
        tuple(resolved_conflicts),
    )


def _label_annotation_count(label_path: Path) -> int:
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_label_class_ids(label_path: Path) -> list[int]:
    class_ids: list[int] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO annotation at {label_path}:{line_number}: {raw_line!r}.")
        try:
            class_value = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid class id at {label_path}:{line_number}: {parts[0]!r}.") from exc
        class_id = int(class_value)
        if class_value != class_id:
            raise ValueError(f"Class id must be an integer at {label_path}:{line_number}: {parts[0]!r}.")
        class_ids.append(class_id)
    return class_ids


T = TypeVar("T")


def split_items(
    items: Sequence[T],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, tuple[T, ...]]:
    if not items:
        raise ValueError("Cannot split an empty dataset.")
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("`train_ratio` must be positive and `val_ratio` must be non-negative.")
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("`train_ratio + val_ratio` must be less than or equal to 1.")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    raw_counts = {name: len(shuffled) * ratio for name, ratio in ratios.items()}
    counts = {name: int(raw_counts[name]) for name in ratios}
    remaining = len(shuffled) - sum(counts.values())
    for name in sorted(ratios, key=lambda key: (raw_counts[key] - counts[key], ratios[key]), reverse=True):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": tuple(shuffled[:train_end]),
        "val": tuple(shuffled[train_end:val_end]),
        "test": tuple(shuffled[val_end:]),
    }


def build_energy_dataset_plan(
    *,
    source_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> EnergyDatasetPlan:
    items, categories_by_id, class_counts, unmatched_label_paths, resolved_conflicts = discover_energy_items(source_root)
    class_config = build_class_config(categories_by_id, set(class_counts))
    splits = split_items(items, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return EnergyDatasetPlan(
        source_root=source_root.resolve(),
        output_root=output_root.resolve(),
        categories_by_id=categories_by_id,
        class_config=class_config,
        class_counts=class_counts,
        unmatched_label_paths=unmatched_label_paths,
        splits=splits,
        resolved_label_conflicts=resolved_conflicts,
    )


def prepare_energy_dataset(plan: EnergyDatasetPlan, *, dry_run: bool, force: bool) -> None:
    if dry_run:
        return

    if plan.output_root.exists() and _has_existing_output(plan.output_root) and not force:
        raise FileExistsError(
            f"Output directory already contains files: {plan.output_root}. "
            "Re-run with --force to replace managed energy dataset files."
        )
    if force:
        _remove_managed_output(plan.output_root)

    for split_name in ("train", "val", "test"):
        for section in ("images", "labels", "geo"):
            (plan.output_root / section / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, split_items_ in plan.splits.items():
        for item in split_items_:
            _copy_item_into_split(plan.output_root, split_name, item, overwrite=True)

    (plan.output_root / "classes.txt").write_text(
        "\n".join(plan.class_config.classes) + "\n",
        encoding="utf-8",
    )
    (plan.output_root / "metadata.json").write_text(
        json.dumps(_metadata(plan), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (plan.output_root / "README.md").write_text(_render_readme(plan), encoding="utf-8")


def _copy_item_into_split(output_root: Path, split_name: str, item: EnergySourceItem, *, overwrite: bool) -> None:
    image_target = output_root / "images" / split_name / item.relative_image_path
    label_target = output_root / "labels" / split_name / item.relative_label_path
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and image_target.exists():
        raise FileExistsError(f"Refusing to overwrite existing image: {image_target}")
    shutil.copy2(item.image_path, image_target)
    if item.label_path is None:
        label_target.write_text("", encoding="utf-8")
    else:
        shutil.copy2(item.label_path, label_target)

    for geo_path in item.geo_paths:
        geo_target = output_root / "geo" / split_name / item.relative_image_path.with_suffix(geo_path.suffix)
        geo_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(geo_path, geo_target)


def load_existing_items(output_root: Path) -> dict[str, tuple[EnergySourceItem, ...]]:
    existing: dict[str, tuple[EnergySourceItem, ...]] = {}
    for split_name in ("train", "val", "test"):
        images_dir = output_root / "images" / split_name
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Prepared split directory does not exist: {images_dir}.")
        split_items: list[EnergySourceItem] = []
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = output_root / "labels" / split_name / image_path.with_suffix(".txt").name
            geo_paths = tuple(
                geo_path
                for suffix in GEO_SUFFIXES
                if (geo_path := output_root / "geo" / split_name / image_path.with_suffix(suffix).name).exists()
            )
            split_items.append(
                EnergySourceItem(
                    project_name="<existing>",
                    image_path=image_path,
                    label_path=label_path if label_path.is_file() else None,
                    geo_paths=geo_paths,
                )
            )
        existing[split_name] = tuple(split_items)
    return existing


def build_energy_merge_plan(
    *,
    source_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> EnergyDatasetPlan:
    output_root = output_root.resolve()
    metadata_path = output_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Merge requires an existing prepared dataset with metadata.json: {metadata_path}. "
            "Run without --merge first."
        )
    classes_path = output_root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Merge requires an existing prepared dataset with classes.txt: {classes_path}.")
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    existing_classes = tuple(
        line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )

    new_items, categories_by_id, new_class_counts, unmatched_label_paths, resolved_conflicts = discover_energy_items(source_root)
    class_config = build_class_config(categories_by_id, set(new_class_counts))
    if class_config.class_id_map is not None or tuple(class_config.classes) != existing_classes:
        raise ValueError(
            "Merged export classes must match the prepared dataset classes.\n"
            f"  prepared: {list(existing_classes)}\n"
            f"  merged:   {list(class_config.classes)}"
        )

    existing_splits = load_existing_items(output_root)
    existing_names = {item.relative_image_path.as_posix() for items in existing_splits.values() for item in items}
    collisions = sorted(existing_names & {item.relative_image_path.as_posix() for item in new_items})
    if collisions:
        lines = ["Merged export image names collide with the prepared dataset:"]
        lines.extend(f"  - {name}" for name in collisions[:20])
        if len(collisions) > 20:
            lines.append(f"  - ... {len(collisions) - 20} more")
        raise ValueError("\n".join(lines))

    added_splits = split_items(new_items, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    splits = {name: existing_splits[name] + added_splits[name] for name in ("train", "val", "test")}

    class_counts: Counter[int] = Counter()
    for split_items_ in existing_splits.values():
        for item in split_items_:
            if item.label_path is not None:
                class_counts.update(read_label_class_ids(item.label_path))
    class_counts.update(new_class_counts)

    return EnergyDatasetPlan(
        source_root=Path(str(old_metadata.get("source_root", output_root))),
        output_root=output_root,
        categories_by_id=categories_by_id,
        class_config=class_config,
        class_counts=class_counts,
        unmatched_label_paths=unmatched_label_paths,
        splits=splits,
        merge_source_roots=tuple(Path(path) for path in old_metadata.get("merge_source_roots", []))
        + (Path(source_root).resolve(),),
        preserved_unmatched_labels=tuple(str(path) for path in old_metadata.get("unmatched_labels", [])),
        preserved_missing_label_images=tuple(str(path) for path in old_metadata.get("missing_label_images", [])),
        preserved_resolved_label_conflicts=tuple(
            dict(entry) for entry in old_metadata.get("resolved_label_conflicts", [])
        ),
        resolved_label_conflicts=resolved_conflicts,
        pruned_unlabeled=old_metadata.get("pruned_unlabeled"),
        added_splits=added_splits,
    )


def merge_energy_dataset(plan: EnergyDatasetPlan, *, dry_run: bool) -> None:
    if dry_run:
        return
    if plan.added_splits is None:
        raise ValueError("Merge plan does not carry added splits; rebuild it with build_energy_merge_plan.")

    for split_name, split_items_ in plan.added_splits.items():
        for item in split_items_:
            _copy_item_into_split(plan.output_root, split_name, item, overwrite=False)

    (plan.output_root / "classes.txt").write_text(
        "\n".join(plan.class_config.classes) + "\n",
        encoding="utf-8",
    )
    (plan.output_root / "metadata.json").write_text(
        json.dumps(_metadata(plan), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (plan.output_root / "README.md").write_text(_render_readme(plan), encoding="utf-8")


def find_empty_label_items(output_root: Path) -> dict[str, tuple[EnergySourceItem, ...]]:
    """Return prepared-dataset items whose label file exists but has no annotations."""
    empty_by_split: dict[str, tuple[EnergySourceItem, ...]] = {}
    for split_name, items in load_existing_items(output_root).items():
        empty_by_split[split_name] = tuple(
            item for item in items if item.label_path is not None and item.label_path.stat().st_size == 0
        )
    return empty_by_split


def prune_empty_label_items(*, dataset_root: Path, output_root: Path, dry_run: bool) -> dict[str, object]:
    """Move images without annotations out of a prepared dataset into a holding root.

    The holding root keeps a flat `images/`, `labels/`, and `geo/` layout plus a
    `manifest.json` with the original split of each image, so the pool can be
    annotated in Label Studio and merged back into the dataset later.
    """
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    empty_by_split = find_empty_label_items(dataset_root)
    pruned_items = [item for items in empty_by_split.values() for item in items]

    summary: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "pruned_images": len(pruned_items),
        "pruned_by_split": {name: len(items) for name, items in empty_by_split.items() if items},
    }
    if dry_run or not pruned_items:
        return summary

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Prune output root already exists and is not empty: {output_root}")

    for section in ("images", "labels", "geo"):
        (output_root / section).mkdir(parents=True, exist_ok=True)

    manifest_items: dict[str, dict[str, object]] = {}
    for split_name, items in empty_by_split.items():
        for item in items:
            image_name = item.relative_image_path.name
            shutil.move(str(item.image_path), output_root / "images" / image_name)
            if item.label_path is not None:
                shutil.move(str(item.label_path), output_root / "labels" / item.relative_label_path.name)
            moved_geo = []
            for geo_path in item.geo_paths:
                geo_name = item.relative_image_path.with_suffix(geo_path.suffix).name
                shutil.move(str(geo_path), output_root / "geo" / geo_name)
                moved_geo.append(geo_name)
            manifest_items[image_name] = {"split": split_name, "geo": moved_geo}

    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "total": len(pruned_items),
                "items": manifest_items,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        _render_prune_readme(dataset_root, len(pruned_items)), encoding="utf-8"
    )

    _rewrite_metadata_after_prune(dataset_root, output_root, len(pruned_items))
    (dataset_root / "README.md").write_text(
        _render_readme_from_state(dataset_root), encoding="utf-8"
    )
    return summary


def summarize_plan(plan: EnergyDatasetPlan) -> str:
    lines = [
        f"Source root: {plan.source_root}",
        f"Output root: {plan.output_root}",
        f"Images: {plan.total_images}",
        f"Labels: {plan.total_labels}",
        f"Unmatched labels skipped: {len(plan.unmatched_label_paths)}",
        f"Missing labels to be written as empty files: {plan.missing_labels}",
        f"Empty source labels: {plan.empty_labels}",
        "Splits:",
    ]
    for split_name in ("train", "val", "test"):
        lines.append(f"  - {split_name}: {len(plan.splits[split_name])}")

    lines.append("Classes:")
    for class_id, name in sorted(plan.categories_by_id.items()):
        lines.append(f"  - {class_id}: {name} ({plan.class_counts.get(class_id, 0)} annotations)")

    if plan.class_config.class_id_map is None:
        lines.append("class_id_map: not needed")
    else:
        lines.append(f"class_id_map: {list(plan.class_config.class_id_map)}")

    geo_counts = plan.geo_counts
    lines.append(f"Geo files: .jgw={geo_counts.get('.jgw', 0)}, .prj={geo_counts.get('.prj', 0)}")
    if plan.merge_source_roots:
        lines.append("Merged source roots:")
        lines.extend(f"  - {path}" for path in plan.merge_source_roots)
    unmatched_labels = plan.all_unmatched_labels
    if unmatched_labels:
        lines.append("Unmatched label files:")
        lines.extend(f"  - {path}" for path in unmatched_labels[:10])
        if len(unmatched_labels) > 10:
            lines.append(f"  - ... {len(unmatched_labels) - 10} more")
    missing_label_images = plan.all_missing_label_images
    if missing_label_images:
        lines.append("Images without source label files:")
        lines.extend(f"  - {path}" for path in missing_label_images[:10])
        if len(missing_label_images) > 10:
            lines.append(f"  - ... {len(missing_label_images) - 10} more")
    conflicts = plan.all_resolved_label_conflicts
    if conflicts:
        lines.append("Resolved duplicate-label conflicts (kept the file with the most annotations):")
        for conflict in conflicts[:10]:
            lines.append(f"  - {conflict['image_stem']}: kept {conflict['kept_label_path']}")
        if len(conflicts) > 10:
            lines.append(f"  - ... {len(conflicts) - 10} more")
    return "\n".join(lines)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", default="data/energy2", help="Source Label Studio YOLO export root.")
    parser.add_argument("--output-root", default="data/energy_obj", help="Prepared dataset output root.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Fraction of images for training.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of images for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without writing files.")
    parser.add_argument("--force", action="store_true", help="Replace managed files under the output root.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge the source export into the existing prepared dataset instead of replacing it.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare-energy-dataset",
        description="Prepare the local energy object detection dataset for the MMDetection YOLOTxtDataset.",
    )
    configure_parser(parser)
    args = parser.parse_args(argv)
    if args.merge:
        plan = build_energy_merge_plan(
            source_root=Path(args.source_root),
            output_root=Path(args.output_root),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(summarize_plan(plan))
        if args.dry_run:
            print("Dry run only; no files were written.")
            return 0
        merge_energy_dataset(plan, dry_run=False)
        print(f"Merged dataset updates written to: {plan.output_root}")
        return 0
    plan = build_energy_dataset_plan(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(summarize_plan(plan))
    if args.dry_run:
        print("Dry run only; no files were written.")
        return 0
    prepare_energy_dataset(plan, dry_run=False, force=args.force)
    print(f"Prepared dataset written to: {plan.output_root}")
    return 0


def _project_dirs(source_root: Path) -> tuple[Path, ...]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}.")
    return tuple(sorted(path for path in source_root.iterdir() if path.is_dir()))


def _validate_flattened_names(items: Sequence[EnergySourceItem]) -> None:
    by_name: dict[str, list[Path]] = {}
    for item in items:
        by_name.setdefault(item.relative_image_path.as_posix(), []).append(item.image_path)

    collisions = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if not collisions:
        return

    lines = ["Flattened energy dataset would overwrite files with the same name:"]
    for name, paths in sorted(collisions.items()):
        lines.append(f"  - {name}")
        lines.extend(f"    - {path}" for path in paths)
    raise ValueError("\n".join(lines))


def _has_existing_output(output_root: Path) -> bool:
    return any((output_root / name).exists() for name in MANAGED_OUTPUT_NAMES)


def _remove_managed_output(output_root: Path) -> None:
    for name in MANAGED_OUTPUT_NAMES:
        path = output_root / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _metadata(plan: EnergyDatasetPlan) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source_root": str(plan.source_root),
        "output_root": str(plan.output_root),
        "total_images": plan.total_images,
        "total_labels": plan.total_labels,
        "unmatched_labels": list(plan.all_unmatched_labels),
        "missing_label_images": list(plan.all_missing_label_images),
        "missing_labels": plan.missing_labels,
        "empty_labels": plan.empty_labels,
        "splits": {name: len(items) for name, items in plan.splits.items()},
        "categories": [
            {"id": class_id, "name": name, "annotations": plan.class_counts.get(class_id, 0)}
            for class_id, name in sorted(plan.categories_by_id.items())
        ],
        "classes": list(plan.class_config.classes),
        "class_id_map": list(plan.class_config.class_id_map) if plan.class_config.class_id_map else None,
        "merge_source_roots": [str(path) for path in plan.merge_source_roots],
        "resolved_label_conflicts": list(plan.all_resolved_label_conflicts),
        "geo_files": dict(plan.geo_counts),
    }
    if plan.pruned_unlabeled is not None:
        metadata["pruned_unlabeled"] = plan.pruned_unlabeled
    return metadata


def _render_readme_from_state(dataset_root: Path) -> str:
    """Re-render the dataset README from the current on-disk state.

    Reads preserved provenance (merge roots, unmatched labels, conflicts,
    pruned pool) from `metadata.json`, so call it after the metadata file
    already reflects the latest operation.
    """
    metadata_path = dataset_root / "metadata.json"
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}

    existing = load_existing_items(dataset_root)
    classes = [
        line.strip()
        for line in (dataset_root / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    class_counts: Counter[int] = Counter()
    for items in existing.values():
        for item in items:
            if item.label_path is not None:
                class_counts.update(read_label_class_ids(item.label_path))

    class_id_map = old_metadata.get("class_id_map")
    plan = EnergyDatasetPlan(
        source_root=Path(str(old_metadata.get("source_root", dataset_root))),
        output_root=dataset_root,
        categories_by_id={class_id: name for class_id, name in enumerate(classes)},
        class_config=DatasetClassConfig(
            classes=tuple(classes),
            class_id_map=tuple(int(class_id) for class_id in class_id_map) if class_id_map else None,
        ),
        class_counts=class_counts,
        unmatched_label_paths=(),
        splits=existing,
        merge_source_roots=tuple(Path(path) for path in old_metadata.get("merge_source_roots", [])),
        preserved_unmatched_labels=tuple(str(path) for path in old_metadata.get("unmatched_labels", [])),
        preserved_missing_label_images=tuple(
            str(path) for path in old_metadata.get("missing_label_images", [])
        ),
        preserved_resolved_label_conflicts=tuple(
            dict(entry) for entry in old_metadata.get("resolved_label_conflicts", [])
        ),
        pruned_unlabeled=old_metadata.get("pruned_unlabeled"),
    )
    return _render_readme(plan)


def _rewrite_metadata_after_prune(dataset_root: Path, output_root: Path, pruned_count: int) -> None:
    metadata_path = dataset_root / "metadata.json"
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}

    existing = load_existing_items(dataset_root)
    class_counts: Counter[int] = Counter()
    total_labels = 0
    geo_counts: Counter[str] = Counter()
    for items in existing.values():
        for item in items:
            if item.label_path is not None:
                total_labels += 1
                class_counts.update(read_label_class_ids(item.label_path))
            for geo_path in item.geo_paths:
                geo_counts[geo_path.suffix.lower()] += 1

    classes = [
        line.strip() for line in (dataset_root / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    metadata = dict(old_metadata)
    metadata.update(
        {
            "total_images": sum(len(items) for items in existing.values()),
            "total_labels": total_labels,
            "missing_label_images": [],
            "missing_labels": 0,
            "empty_labels": 0,
            "splits": {name: len(items) for name, items in existing.items()},
            "categories": [
                {"id": class_id, "name": name, "annotations": class_counts.get(class_id, 0)}
                for class_id, name in enumerate(classes)
            ],
            "geo_files": dict(geo_counts),
            "pruned_unlabeled": {
                "root": str(output_root),
                "images": pruned_count,
                "manifest": str(output_root / "manifest.json"),
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _render_prune_readme(dataset_root: Path, pruned_count: int) -> str:
    return f"""# Unlabeled images pruned from `{_display_path(dataset_root)}`

{pruned_count} images without annotations (empty YOLO label files) were moved here from the prepared dataset. They are waiting for annotation in Label Studio.

## Structure

```text
images/       pruned image tiles (flat)
labels/       the original empty .txt files kept for reference
geo/          corresponding .jgw world files and .prj projections
manifest.json original split of each image inside the prepared dataset
```

## Workflow

1. Import `images/` into Label Studio and annotate the objects.
2. Export the project as YOLO and merge it back into the prepared dataset:

```bash
uv run python scripts/prepare_energy_dataset.py --merge --source-root <export root> --dry-run
uv run python scripts/prepare_energy_dataset.py --merge --source-root <export root>
```
"""


def _render_readme(plan: EnergyDatasetPlan) -> str:
    class_lines = "\n".join(
        f"- `{class_id}`: `{name}` ({plan.class_counts.get(class_id, 0)} annotations)"
        for class_id, name in sorted(plan.categories_by_id.items())
    )
    split_lines = "\n".join(f"- `{name}`: {len(items)} images" for name, items in plan.splits.items())
    geo_counts = plan.geo_counts
    unmatched_lines = "\n".join(f"- `{_display_path(Path(path))}`" for path in plan.all_unmatched_labels) or "- none"
    missing_label_lines = (
        "\n".join(f"- `{_display_path(Path(path))}`" for path in plan.all_missing_label_images) or "- none"
    )
    merged_lines = ""
    if plan.merge_source_roots:
        merged_lines = "\n\nMerged Label Studio YOLO exports:\n\n" + "\n".join(
            f"- `{_display_path(path)}`" for path in plan.merge_source_roots
        )
    split_note = (
        "The split is deterministic and was generated with `--seed 42`."
        if not plan.merge_source_roots
        else "Existing images kept their splits; the merged images were split deterministically with `--seed 42`."
    )
    conflict_lines = (
        "\n".join(
            f"- `{conflict['image_stem']}`: kept `{_display_path(Path(str(conflict['kept_label_path'])))}`"
            f" ({len(conflict['rejected_label_paths'])} rejected)"
            for conflict in plan.all_resolved_label_conflicts
        )
        or "- none"
    )
    pruned_section = ""
    if plan.pruned_unlabeled is not None:
        pruned_root = Path(str(plan.pruned_unlabeled.get("root", "")))
        pruned_count = plan.pruned_unlabeled.get("images", 0)
        pruned_section = (
            "\n## Pruned Unlabeled Images\n\n"
            f"{pruned_count} images without annotations were moved to `{_display_path(pruned_root)}` "
            "for re-annotation and are not part of the dataset splits. See its `README.md` and `manifest.json`.\n"
        )
    class_id_map = (
        "not needed because source class ids are contiguous"
        if plan.class_config.class_id_map is None
        else ", ".join(str(class_id) for class_id in plan.class_config.class_id_map)
    )
    return f"""# energy_obj

Prepared local object detection dataset built from Label Studio YOLO exports in `{_display_path(plan.source_root)}`.{merged_lines}

## Structure

```text
data/energy_obj/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── geo/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt
├── metadata.json
└── README.md
```

Each split is intentionally flat: images, labels, and geo files are written directly under `images/<split>/`, `labels/<split>/`, and `geo/<split>/`. Label files are renamed to match image stems, so the project `YOLOTxtDataset` can resolve labels by relative image path.

## Splits

{split_lines}

{split_note}

## Classes

{class_lines}

`class_id_map`: {class_id_map}.

## Unmatched Labels

Labels that do not have a corresponding source image are not copied into the prepared dataset:

{unmatched_lines}

Images without a corresponding source label get an empty `.txt` file:

{missing_label_lines}

## Resolved Duplicate-Label Conflicts

When multiple source label files match one image, the file with the most annotation rows is kept; the rest are rejected:

{conflict_lines}

## Geo Files

World files and projection files are copied into the flat `geo/<split>/` tree:

- `.jgw`: {geo_counts.get(".jgw", 0)}
- `.prj`: {geo_counts.get(".prj", 0)}
{pruned_section}
## Rebuild

```bash
uv run python scripts/prepare_energy_dataset.py --dry-run
uv run python scripts/prepare_energy_dataset.py --force
```

Merge an additional Label Studio YOLO export into the prepared dataset:

```bash
uv run python scripts/prepare_energy_dataset.py --merge --source-root <export root> --dry-run
uv run python scripts/prepare_energy_dataset.py --merge --source-root <export root>
```

## Training

```bash
uv run python scripts/train.py --config configs/project/yolov3_d53_energy_obj_mmdet.toml --dry-run
uv run python scripts/train.py --config configs/project/yolov3_d53_energy_obj_mmdet.toml
```

## TensorBoard

```bash
uv run tensorboard --logdir log/tensorboard/yolov3_d53_energy_obj_mmdet --port 6006
```

## Inference

```bash
uv run python scripts/infer.py \\
  --config configs/project/yolov3_d53_energy_obj_mmdet.toml \\
  --checkpoint checkpoints/yolov3_d53_energy_obj_mmdet/epoch_10.pth \\
  --input data/energy_obj/images/test \\
  --output-dir outputs/yolov3_d53_energy_obj_mmdet_test
```
"""


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)
