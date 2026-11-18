# Project wrapper: Cascade R-CNN r50 FPN with the mmdet ms-3x training recipe
# (multiscale 640-1333 + RepeatDataset x3 + 12-outer-epoch schedule, LR milestones [9, 11]).
# mmdet 3.3.0 ships ms-3x only for cascade-mask variants; for bbox-only detection we
# combine the common ms_3x_coco recipe with the cascade-rcnn_r50_fpn model definition.
# Constraints that shaped this file (mmengine 0.10.7):
#   - any top-level import makes mmengine treat the config as lazy-import and reject
#     a plain `_base_` assignment;
#   - `_base_` is eval'd in isolation, so no intermediate variables — literal paths only.
# Hence literal paths relative to this file into the project venv. If the venv is
# rebuilt at a different Python version, update the two lines below.
# Datasets/evaluators are replaced by the project pipeline (YOLOTxtDataset +
# YOLOTxtMetric) at build time.

_base_ = [
    '../../.venv/lib/python3.10/site-packages/mmdet/.mim/configs/common/ms_3x_coco.py',
    '../../.venv/lib/python3.10/site-packages/mmdet/.mim/configs/_base_/models/cascade-rcnn_r50_fpn.py',
]
