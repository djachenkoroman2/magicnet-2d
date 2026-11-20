# AGENTS.md

## Запуск обучения

- При каждом запуске обучения также запускать TensorBoard (если ещё не запущен) в фоне:

  ```bash
  nohup uv run tensorboard --logdir_spec "mmdet:log/tensorboard,yolo:log/yolo" --port 6006 --host 0.0.0.0 > /tmp/opencode/tensorboard.log 2>&1 &
  ```

- Логи: MMDetection → `log/tensorboard/`, Ultralytics YOLO → `log/yolo/`.
- Обучение запускается в фоне (`nohup`), сериализовано (одна GPU).
