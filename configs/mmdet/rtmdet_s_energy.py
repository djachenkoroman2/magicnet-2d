# Project wrapper: RTMDet-s (CSPNeXt + CSPNeXtPAFPN, anchor-free one-stage) with
# the native 300-epoch recipe, adapted to the tiled energy_obj dataset at 1280
# native tile resolution. RTMDet is the modern one-stage anchor-free detector in
# mmdet (the "YOLO-successor" line, with CachedMosaic + CachedMixUp + EMA) and is
# evaluated with the same YOLOTxtMetric as the two-stage leaders, so the numbers
# are directly comparable. See doc/experiments.md, priority "modern architectures".
#
# Base: mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py. We keep the model + schedule
# and only widen the pipeline from 640 to 1280 so the 50-165 px objects in the
# 1280x1280 tiles keep full resolution (COCO 640 would shrink them x0.5 — the same
# lever that already hurt YOLOv3-608). CachedMosaic / CachedMixUp / Pad / RandomCrop
# are all resized to 1280.
#
# Constraints mirror configs/mmdet/cascade-rcnn_r50_fpn_ms3x_energy.py:
#   - `_base_` eval'd in isolation -> literal paths only;
#   - top-level import makes the config lazy-import and rejects plain `_base_`.
# Datasets/evaluators are replaced by the project pipeline (YOLOTxtDataset +
# YOLOTxtMetric) at build time.

_base_ = [
    '../../.venv/lib/python3.10/site-packages/mmdet/.mim/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(1280, 1280), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1280, 1280)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(1280, 1280),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1280, 1280)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
