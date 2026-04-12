# MagicNet 2D

Локальный шаблон проекта для обучения и инференса моделей детекции объектов в 2D на базе PyTorch, MMDetection, TensorBoard, `uv` и с готовым smoke-test сценарием для `Ultralytics YOLO`.

## Архитектура

Проект разделён на пять слоёв:

1. `configs/project/default.toml` хранит прикладную конфигурацию проекта: пути, описание датасета, выбранный конфиг MMDetection, параметры обучения и инференса.
2. `src/magicnet_2d/` содержит Python-пакет с CLI, подготовкой рантайм-конфига MMDetection, валидацией путей и диагностикой окружения.
3. `scripts/` предоставляет простые точки входа, которые можно запускать через `uv run python ...`.
4. `log/`, `checkpoints/` и `outputs/` разделяют TensorBoard-логи, чекпойнты и результаты инференса.
5. `configs/yolo/` и `data/yolo/coco8_ultralytics/` добавляют готовый небольшой датасет и конфиги для проверки YOLO-пайплайна.

Во время работы проект создаёт:

- MMDetection runtime-логи в `log/<run_name>/`;
- MMDetection TensorBoard-логи в `log/tensorboard/<run_name>/`;
- локальные визуализации MMDetection в `log/local/<run_name>/`;
- чекпойнты MMDetection в `checkpoints/<run_name>/`;
- логи и артефакты YOLO в `log/yolo/<run_name>/` и `checkpoints/yolo/<run_name>/`.

## Структура проекта

```text
magicnet-2d/
├── checkpoints/
├── configs/
│   ├── mmdet/
│   │   └── README.md
│   ├── project/
│   │   └── default.toml
│   └── yolo/
│       ├── README.md
│       ├── coco8_smoke.yaml
│       └── coco8_ultralytics.yaml
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
├── pyproject.toml
├── uv.lock
└── README.md
```

## Точки входа

У проекта есть два равноправных способа запуска:

- обёртки в `scripts/`, например `uv run python scripts/train.py ...`;
- единый CLI пакета, например `uv run magicnet train ...`.

В этом README для наглядности используются команды через `scripts/`.

## Установка

### 1. Подготовка Python и базового окружения

```bash
uv python install 3.10
uv sync --extra dev
```

`uv sync` создаёт и синхронизирует локальную `.venv`. Если вы не активировали её явно, запускайте проектные команды через `uv run ...`, чтобы не использовать другой `python` из Conda или системного окружения.

### 2. Подбор совместимого профиля под железо

```bash
python scripts/check_env.py check
```

Скрипт:

- определяет ОС, архитектуру, Python, наличие NVIDIA GPU, драйвер, CUDA и cuDNN;
- выбирает рекомендуемый профиль (`cpu`, `cu118`, `cu121`);
- печатает безопасный план установки;
- поддерживает `install --execute`, если вы хотите выполнить только безопасные автоматизируемые шаги.

### 3. MMDetection на CPU

```bash
uv sync --extra cpu --extra dev
uv run mim install mmcv==2.1.0
```

`mmdet` для типовых детекторов вроде Faster R-CNN требует compiled ops из полноценного `mmcv`, поэтому `uv sync` нужно дополнять установкой `mmcv` через OpenMIM.

Если вы повторно запускаете `uv sync` для окружения с MMDetection, после синхронизации снова выполните:

```bash
uv run mim install mmcv==2.1.0
```

### 4. YOLO smoke-test окружение

```bash
uv sync --extra yolo --extra dev
```

### 5. Совмещённое окружение MMDetection + YOLO

Если вы хотите запускать оба пайплайна в одной `.venv`, синхронизируйте сразу оба extras:

```bash
uv sync --extra cpu --extra yolo --extra dev
uv run mim install mmcv==2.1.0
```

### 6. GPU-установка

Сначала проверьте, что система действительно видит NVIDIA GPU:

```bash
python scripts/check_env.py check
```

Если нужен автоматический подбор и установка поддерживаемого GPU-профиля:

```bash
python scripts/install_system_deps.py install --execute
```

Если вы уже знаете нужный профиль, можно указать его явно:

```bash
python scripts/install_system_deps.py install --profile cu121 --execute
```

или

```bash
python scripts/install_system_deps.py install --profile cu118 --execute
```

Скрипт не пытается автоматически ставить драйвер NVIDIA или системные пакеты с root-доступом. Вместо этого он даёт совместимые рекомендации и выполняет только безопасные команды уровня пользователя.

## Подготовка датасета

### MMDetection dataset

По умолчанию шаблон ожидает COCO-совместимую структуру:

```text
data/coco/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
└── val/
```

Текущий `configs/project/default.toml` также содержит демонстрационный placeholder-класс:

```toml
[dataset]
classes = ["object"]
```

Перед реальным обучением обновите `paths.data_root`, `dataset.classes` и пути к аннотациям под ваш датасет.

### YOLO smoke-test dataset

Для быстрого теста Ultralytics YOLO в репозитории используется официальный мини-датасет COCO8:

- датасет: `data/yolo/coco8_ultralytics`
- dataset config: `configs/yolo/coco8_ultralytics.yaml`
- training config: `configs/yolo/coco8_smoke.yaml`

Если нужно скачать его заново:

```bash
uv run python scripts/download_yolo_coco8.py --force
```

Smoke-test конфиг использует `yolov8n.yaml` и обучает модель с нуля, чтобы не зависеть от скачивания внешних pretrained-весов.

Важно: `data/yolo/coco8_ultralytics` используется только YOLO-пайплайном и не подходит для `scripts/train.py` без отдельной перенастройки MMDetection-конфига и путей.

## Обучение и инференс с MMDetection

### Проверка конфигурации без запуска обучения

```bash
uv run python scripts/train.py --config configs/project/default.toml --dry-run
```

Dry-run:

- валидирует пути к датасету;
- разрешает ссылку `mmdet::...` в реальный upstream-конфиг;
- собирает `resolved_config.py` в `log/<run_name>/`.

### Запуск обучения

```bash
uv run python scripts/train.py --config configs/project/default.toml
```

Если GPU доступен и CUDA-окружение установлено корректно, MMDetection будет использовать его через обычный PyTorch runtime. Для выбора конкретной карты удобно использовать `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config configs/project/default.toml
```

### Переопределение имени запуска и upstream-конфига

```bash
uv run python scripts/train.py \
  --config configs/project/default.toml \
  --run-name my_experiment \
  --mmdet-config mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
```

### Инференс

На одном изображении:

```bash
uv run python scripts/infer.py \
  --config configs/project/default.toml \
  --input path/to/image.jpg \
  --checkpoint checkpoints/my_experiment/epoch_12.pth
```

На директории:

```bash
uv run python scripts/infer.py \
  --config configs/project/default.toml \
  --input path/to/images \
  --checkpoint checkpoints/my_experiment/epoch_12.pth \
  --output-dir outputs/batch_inference
```

Явный запуск на GPU:

```bash
uv run python scripts/infer.py \
  --config configs/project/default.toml \
  --input path/to/image.jpg \
  --checkpoint checkpoints/my_experiment/epoch_12.pth \
  --device cuda:0
```

Результаты сохраняются в указанную папку инференса, а агрегированный JSON-отчёт записывается в `predictions.json`.

## YOLO Smoke-Test Обучение и Инференс

### Проверка конфигурации без запуска

```bash
uv run python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml --dry-run
```

### Обучение на CPU

```bash
uv run python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml
```

### Обучение на GPU

```bash
uv run python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml --device 0
```

Если карт несколько, можно указать, например, `--device 0,1`.

После завершения:

- runtime-логи и TensorBoard events будут в `log/yolo/coco8_smoke/`;
- исходные Ultralytics weights будут в `log/yolo/coco8_smoke/weights/`;
- копии `best.pt` и `last.pt` будут в `checkpoints/yolo/coco8_smoke/`.

### YOLO-инференс

На произвольной фотографии:

```bash
uv run python scripts/yolo_infer.py \
  --weights checkpoints/yolo/coco8_smoke/best.pt \
  --source path/to/photo.jpg \
  --output-dir outputs/yolo/custom_photo
```

На GPU:

```bash
uv run python scripts/yolo_infer.py \
  --weights checkpoints/yolo/coco8_smoke/best.pt \
  --source path/to/photo.jpg \
  --output-dir outputs/yolo/custom_photo \
  --device 0
```

На директории с изображениями:

```bash
uv run python scripts/yolo_infer.py \
  --weights checkpoints/yolo/coco8_smoke/best.pt \
  --source path/to/images \
  --output-dir outputs/yolo/batch_predict
```

Аннотированные изображения и `.txt`-предсказания сохраняются в указанную папку внутри `outputs/yolo/`.

## TensorBoard

Все TensorBoard-логи сохраняются в `log/`.

MMDetection:

```bash
uv run tensorboard --logdir log/tensorboard --port 6006
```

YOLO по всем запускам:

```bash
uv run tensorboard --logdir log/yolo --port 6006
```

YOLO по одному конкретному smoke-test запуску:

```bash
uv run tensorboard --logdir log/yolo/coco8_smoke --port 6006
```

## Конфигурация MMDetection

В `configs/project/default.toml` используется ссылка вида:

```toml
[mmdet]
config = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
```

Префикс `mmdet::` означает: возьми конфиг из установленного каталога конфигов MMDetection:

```text
<site-packages>/mmdet/.mim/configs/
```

Это позволяет не хранить в репозитории большой набор upstream-конфигов, но всё равно использовать стандартные модели MMDetection.

Проектный TOML-конфиг живёт отдельно от upstream-конфига и накладывает на него:

- локальные пути к данным;
- список классов;
- директории логов и чекпойнтов;
- настройки TensorBoard и локальной визуализации;
- параметры инференса и dry-run диагностики.

## Тесты и базовая проверка

Юнит-тесты:

```bash
python -m pytest -q
```

Проверка диагностики окружения:

```bash
python scripts/check_env.py check
```

Проверка MMDetection-конфига без обучения:

```bash
uv run python scripts/train.py --config configs/project/default.toml --dry-run
```

Проверка YOLO smoke-test конфига без обучения:

```bash
uv run python scripts/yolo_train.py --config configs/yolo/coco8_smoke.yaml --dry-run
```

## Кратко о решениях

- `uv` используется как основной менеджер окружения и зависимостей.
- `mmcv` устанавливается через OpenMIM, потому что для MMDetection нужны compiled ops, а подходящий wheel зависит от версии PyTorch и CUDA.
- TensorBoard и текстовые логи направляются в `log/`, чтобы не смешивать их с чекпойнтами и выходными предсказаниями.
- Проектный конфиг хранится отдельно от upstream MMDetection-конфига и накладывает на него локальные пути, классы, hook-ы и логи.
- Скрипт установки разделяет безопасную автоматизацию уровня пользователя и ручные шаги для системных компонентов вроде драйвера NVIDIA.
