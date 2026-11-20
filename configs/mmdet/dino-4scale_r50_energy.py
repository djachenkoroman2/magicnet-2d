# Project wrapper: DINO-4scale r50 with the improved 12-epoch recipe, adapted
# to the tiled energy_obj dataset. DINO is a DETR-family end-to-end detector
# (denoising anchor boxes, box refinement) — the most modern architecture
# available in mmdet 3.3.0 and the natural "next step" after the two-stage
# cascade leader. See doc/experiments.md, priority "modern architectures".
#
# Base: mmdet::dino/dino-4scale_r50_improved_8xb2-12e_coco.py (50.1 box AP on
# COCO). We keep the loss / optimizer / denoising recipe and only widen the
# train-resolution schedule so the ~50-165 px objects in the 1280x1280 tiles
# are not downscaled below working range (COCO DETR short-side 480 would shrink
# them x0.375). Scales use the same (w, h) convention as the mmdet Resize
# transform: for a square 1280 tile, scale=(1333, 800) -> 800x800.
#
# Constraints (mmengine 0.10.7) mirror configs/mmdet/cascade-rcnn_r50_fpn_ms3x_energy.py:
#   - `_base_` is eval'd in isolation -> literal paths, no intermediate variables;
#   - any top-level import makes the config lazy-import and rejects plain `_base_`.
# Datasets/evaluators are replaced by the project pipeline (YOLOTxtDataset +
# YOLOTxtMetric) at build time.

_base_ = [
    '../../.venv/lib/python3.10/site-packages/mmdet/.mim/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py',
]

# Train pipeline: single resize branch (the COCO crop branch targets extreme
# aspect ratios, irrelevant for square tiles). Multiscale 800-1024 keeps the
# small objects large while staying within RTX 4090 24GB at batch 2.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 800), (1333, 900), (1333, 1024)],
        keep_ratio=True),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
