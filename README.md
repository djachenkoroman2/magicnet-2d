# MagicNet 2D

Локальный шаблон проекта для обучения и инференса моделей детекции объектов в 2D на базе PyTorch, MMDetection, TensorBoard, `uv` и с готовым smoke-test сценарием для `Ultralytics YOLO`.

## Архитектура

Проект разделён на четыре слоя:

1. `configs/project/default.toml` хранит прикладную конфигурацию проекта: пути, описание датасета, выбранный конфиг MMDetection, параметры обучения и инференса.
2. `src/magicnet_2d/` содержит Python-пакет с CLI, подготовкой рантайм-конфига MMDetection, валидацией путей и безопасной диагностикой окружения.
3. `scripts/` предоставляет удобные точки входа без предварительной установки пакета в editable-режиме.
4. `log/`, `checkpoints/` и `outputs/` разделяют TensorBoard-логи, чекпойнты и результаты инференса.
5. `configs/yolo/` и `data/yolo/coco8_ultralytics/` добавляют готовый небольшой датасет и конфиги для проверки YOLO-пайплайна.

Во время обучения проект создаёт:

- текстовые и runtime-логи в `log/<run_name>/`;
- TensorBoard-логи в `log/tensorboard/<run_name>/`;
- локальные визуализации MMDetection в `log/local/<run_name>/`;
- чекпойнты в `checkpoints/<run_name>/`.

## Структура проекта

```text
magicnet-2d/
├── checkpoints/
├── configs/
│   ├── mmdet/
│   │   └── README.md
│   ├── yolo/
│   │   ├── README.md
│   │   ├── coco8_smoke.yaml
│   │   └── coco8_ultralytics.yaml
│   └── project/
│       └── default.toml
├── data/
│   ├── README.md
│   └── yolo/
│       └── coco8_ultralytics/
├── log/
├── outputs/
├── scripts/
│   ├── check_env.py
│   ├── download_yolo_coco8.py
│   ├── infer.py
│   ├── install_system_deps.py
│   ├── train.py
│   ├── yolo_infer.py
│   └── yolo_train.py
├── src/
│   └── magicnet_2d/
├── tests/
├── .python-version
├── pyproject.toml
└── README.md
```

## Установка

### 1. Подготовка Python и базового окружения

```bash
uv python install 3.10
uv sync --extra dev
```

### 2. Подбор совместимого профиля под железо

```bash
python scripts/check_env.py check
```

Скрипт:

- определяет ОС, архитектуру, Python, наличие NVIDIA GPU, драйвер, CUDA и cuDNN;
- выбирает рекомендуемый профиль (`cpu`, `cu118`, `cu121`);
- печатает безопасный план установки;
- поддерживает `install --execute`, если вы хотите выполнить только безопасные автоматизируемые шаги.

### 3. CPU-only быстрый старт

```bash
uv sync --extra cpu --extra dev
```

### 3a. YOLO smoke-test окружение

```bash
uv sync --extra yolo --extra dev
```

### 4. GPU-установка

```bash
python scripts/install_system_deps.py check
python scripts/install_system_deps.py install --execute
```

Скрипт не пытается автоматически ставить драйвер NVIDIA или системные пакеты с root-доступом. Вместо этого он даёт совместимые рекомендации и выполняет только безопасные команды уровня пользователя.

## Подготовка датасета

По умолчанию шаблон ожидает COCO-совместимую структуру:

```text
data/coco/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
└── val/
```

Измените `configs/project/default.toml`, если ваш датасет хранится в другой структуре или содержит другой набор классов.

## YOLO Smoke-Test Датасет

Для быстрого теста Ultralytics YOLO уже скачан официальный мини-датасет:

- датасет: `data/yolo/coco8_ultralytics`
- dataset config: `configs/yolo/coco8_ultralytics.yaml`
- training config: `configs/yolo/coco8_smoke.yaml`

Если нужно скачать его заново:

```bash
python scripts/download_yolo_coco8.py --force
```

По умолчанию smoke-test использует `yolov8n.pt`. При первом запуске Ultralytics может автоматически скачать веса, если их ещё нет в локальном кеше.

## Обучение

Проверить конфигурацию без запуска обучения:

```bash
python scripts/train.py --config configs/project/default.toml --dry-run
```

Запустить обучение:

```bash
python scripts/train.py --config configs/project/default.toml
```

При необходимости можно переопределить имя запуска и исходный конфиг MMDetection:

```bash
python scripts/train.py \
  --config configs/project/default.toml \
  --run-name my_experiment \
  --mmdet-config mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
```

## Инференс

На одном изображении:

```bash
python scripts/infer.py \
  --config configs/project/default.toml \
  --input path/to/image.jpg \
  --checkpoint checkpoints/my_experiment/epoch_12.pth
```

На директории:

```bash
python scripts/infer.py \
  --config configs/project/default.toml \
  --input path/to/images \
  --checkpoint checkpoints/my_experiment/epoch_12.pth \
  --output-dir outputs/batch_inference
```

Результаты сохраняются в указанную папку инференса, а агрегированный JSON-отчёт записывается в `predictions.json`.

## TensorBoard

Все TensorBoard-логи и связанные лог-файлы сохраняются в `log/`.

Запуск TensorBoard:

```bash
uv run tensorboard --logdir log/tensorboard --port 6006
```

Для YOLO smoke-test удобнее смотреть конкретную ветку логов:

```bash
uv run tensorboard --logdir log/yolo --port 6006
```

## YOLO Smoke-Test Обучение

Запуск обучения:

```bash
python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml
```

Проверка конфигурации без запуска:

```bash
python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml --dry-run
```

После завершения:

- runtime-логи и TensorBoard events будут в `log/yolo/coco8_smoke/`;
- исходные Ultralytics weights будут в `log/yolo/coco8_smoke/weights/`;
- копии `best.pt` и `last.pt` будут в `checkpoints/yolo/coco8_smoke/`.

## YOLO Smoke-Test Инференс

На произвольной фотографии:

```bash
python scripts/yolo_infer.py \
  --weights checkpoints/yolo/coco8_smoke/best.pt \
  --source path/to/photo.jpg \
  --output-dir outputs/yolo/custom_photo
```

На директории с изображениями:

```bash
python scripts/yolo_infer.py \
  --weights checkpoints/yolo/coco8_smoke/best.pt \
  --source path/to/images \
  --output-dir outputs/yolo/batch_predict
```

Аннотированные изображения и `.txt`-предсказания сохраняются в указанную папку внутри `outputs/yolo/`.

## Конфигурация MMDetection

В `configs/project/default.toml` используется ссылка вида:

```toml
[mmdet]
config = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
```

Префикс `mmdet::` означает: возьми конфиг из установленного каталога конфигов MMDetection (`site-packages/mmdet/.mim/configs/`). Это позволяет не хранить в репозитории большой набор upstream-конфигов, но всё равно использовать стандартные модели MMDetection.

## Тесты и базовая проверка

Проверка юнит-тестов:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Проверка диагностики окружения:

```bash
python scripts/check_env.py check
```

## Кратко о решениях

- `uv` используется как основной менеджер окружения и зависимостей.
- TensorBoard и текстовые логи жёстко направляются в `log/`, чтобы не смешивать их с чекпойнтами.
- Проектный конфиг хранится отдельно от MMDetection-конфига и накладывает на него локальные пути, классы, hook-ы и TensorBoard.
- Скрипт установки разделяет безопасную автоматизацию уровня пользователя и ручные шаги для системных компонентов.
