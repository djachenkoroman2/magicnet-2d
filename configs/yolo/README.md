# YOLO Smoke-Test Configs

В этой папке лежат конфиги для быстрого теста пайплайна на `Ultralytics YOLO`.

- `coco8_ultralytics.yaml` — dataset YAML, указывающий на локально скачанный датасет `data/yolo/coco8_ultralytics`.
- `coco8_smoke.yaml` — минимальный training config для локального smoke-test запуска.

Рекомендуемый старт:

```bash
uv sync --extra yolo --extra dev
python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml
```
