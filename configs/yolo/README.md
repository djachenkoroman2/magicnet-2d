# YOLO Smoke-Test Configs

В этой папке лежат конфиги для быстрого теста пайплайна на `Ultralytics YOLO`.

- `coco8_ultralytics.yaml` — dataset YAML, указывающий на локально скачанный датасет `data/yolo/coco8_ultralytics`.
- `coco8_smoke.yaml` — минимальный training config для локального smoke-test запуска.
- `tvdd_ultralytics.yaml` — dataset YAML для локального датасета `data/tvdd`.
- `tvdd_yolov8n.yaml` — базовый training config для обучения `YOLOv8n` на `TVDD`.

Рекомендуемый старт:

```bash
uv sync --extra yolo --extra dev
python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml
```

Для запуска обучения на `TVDD`:

```bash
uv sync --extra yolo --extra dev
uv run python scripts/yolo_train.py --config configs/yolo/tvdd_yolov8n.yaml
```
