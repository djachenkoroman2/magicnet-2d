# Dataset Layout

Шаблон по умолчанию ожидает COCO-совместимый датасет:

```text
data/coco/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
└── val/
```

Если ваши пути отличаются, измените `configs/project/default.toml`.

Для быстрого smoke-test пайплайна YOLO уже скачан локальный мини-датасет:

- `data/yolo/coco8_ultralytics`

Его dataset config находится здесь:

- `configs/yolo/coco8_ultralytics.yaml`
