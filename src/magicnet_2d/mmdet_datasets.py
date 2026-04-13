from __future__ import annotations

from pathlib import Path
from typing import List

import mmcv
from mmengine.fileio import get

from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class YOLOTxtDataset(BaseDetDataset):
    """MMDetection dataset for YOLO txt annotations.

    The dataset expects the usual YOLO layout:

    - images under ``data_prefix['img']``
    - labels under ``ann_file``

    Each label line should follow ``class xc yc w h`` with normalized values.
    """

    IMG_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

    def load_data_list(self) -> List[dict]:
        classes = self._metainfo.get("classes")
        assert classes is not None, "`classes` in `YOLOTxtDataset` can not be None."

        image_root = Path(self.data_prefix["img"])
        label_root = Path(self.ann_file)
        if not image_root.exists():
            raise FileNotFoundError(f"YOLO image directory does not exist: {image_root}")
        if not label_root.exists():
            raise FileNotFoundError(f"YOLO label directory does not exist: {label_root}")

        data_list: list[dict] = []
        for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in self.IMG_SUFFIXES):
            relative_image_path = image_path.relative_to(image_root)
            label_path = label_root / relative_image_path.with_suffix(".txt")

            image_bytes = get(str(image_path), backend_args=self.backend_args)
            image = mmcv.imfrombytes(image_bytes, backend="cv2")
            if image is None:
                raise ValueError(f"Failed to decode image: {image_path}")
            height, width = image.shape[:2]

            data_list.append(
                self.parse_data_info(
                    {
                        "img_id": relative_image_path.as_posix(),
                        "img_path": str(image_path),
                        "label_path": str(label_path),
                        "height": height,
                        "width": width,
                    }
                )
            )
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = {
            "img_id": raw_data_info["img_id"],
            "img_path": raw_data_info["img_path"],
            "height": raw_data_info["height"],
            "width": raw_data_info["width"],
        }

        width = float(raw_data_info["width"])
        height = float(raw_data_info["height"])
        label_path = Path(raw_data_info["label_path"])

        instances: list[dict] = []
        if label_path.exists():
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO annotation at {label_path}:{line_number}: {raw_line!r}")

                class_id = int(float(parts[0]))
                if class_id < 0 or class_id >= len(self._metainfo["classes"]):
                    raise ValueError(f"Unknown class id {class_id} at {label_path}:{line_number}")

                x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
                if box_width <= 0 or box_height <= 0:
                    continue

                x1 = max(0.0, (x_center - box_width / 2.0) * width)
                y1 = max(0.0, (y_center - box_height / 2.0) * height)
                x2 = min(width, (x_center + box_width / 2.0) * width)
                y2 = min(height, (y_center + box_height / 2.0) * height)
                if x2 <= x1 or y2 <= y1:
                    continue

                instances.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_label": class_id,
                        "ignore_flag": 0,
                    }
                )

        data_info["instances"] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get("filter_empty_gt", False) if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get("min_size", 0) if self.filter_cfg is not None else 0

        valid_data_infos: list[dict] = []
        for data_info in self.data_list:
            if filter_empty_gt and not data_info["instances"]:
                continue
            if min(data_info["width"], data_info["height"]) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
