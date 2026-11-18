# Конфиг обучения на тайлах `energy_obj_tiles`

Полный справочник по проектному конфигу `configs/project/energy_obj_tiles_mmdet.toml` — основному конфигу обучения детектора на тайловом датасете `data/energy_obj_tiles/` (см. [dataset.md](dataset.md) про исходный датасет и [training_recommendations.md](training_recommendations.md) про обоснование тайлинга). Составлено 2026-11-16; актуально для текущей структуры проекта.

## Назначение

Конфиг описывает одно обучение MMDetection-детектора на тайловом датасете и весь сопутствующий пайплайн: датасет, длительность обучения, валидацию, чекпойнты, логи и параметры инференса. Парсится `src/magicnet_2d/config.py` (`load_project_config`), исполнение собирается в `src/magicnet_2d/mmdet_support.py` (`build_runtime_config`), запуск — `scripts/train.py`.

Все пути в конфиге относительные — от корня репозитория (корень определяется по `pyproject.toml`).

## Запуск

```bash
# проверка конфигурации без старта обучения
uv run python scripts/train.py --config configs/project/energy_obj_tiles_mmdet.toml --dry-run

# обучение (удобно в tmux, прогон ~6.5 ч на RTX 4090)
tmux new -s train_tiles
uv run python scripts/train.py --config configs/project/energy_obj_tiles_mmdet.toml

# мониторинг
tail -f log/train_tiles_console.log                       # живой вывод (если запуск с tee)
ls log/energy_obj_tiles_mmdet/*/                          # mmengine-логи
uv run tensorboard --logdir log/tensorboard               # метрики

# возобновление прерванного обучения с последнего чекпойнта
# (в конфиге выставить train.resume = "checkpoints/energy_obj_tiles_mmdet/epoch_XX.pth"
#  и перезапустить ту же команду)
```

Куда пишутся результаты (называются по `project.run_name`):

| Что | Куда |
|---|---|
| mmengine-логи, resolved-конфиг, `last_checkpoint` | `log/energy_obj_tiles_mmdet/` |
| локальные scalar-дампы | `log/local/energy_obj_tiles_mmdet/` |
| TensorBoard events | `log/tensorboard/energy_obj_tiles_mmdet/` |
| чекпойнты (`epoch_*.pth`) | `checkpoints/energy_obj_tiles_mmdet/` |
| артефакты оценки (метрики, FP/FN) | `outputs/evaluation/energy_obj_tiles_mmdet/` |

## Справочник по секциям

### `[project]`

| Параметр | Текущее | Описание |
|---|---|---|
| `name` | `magicnet-2d` | Имя проекта (служебное). |
| `run_name` | `energy_obj_tiles_mmdet` | Имя рана. Определяет имена всех выходных директорий (см. таблицу выше). Меняйте при каждом новом осмысленном прогоне, чтобы не перемешивать логи и чекпойнты. Переопределяется флагом `--run-name`. |

### `[paths]`

| Параметр | Текущее | Описание |
|---|---|---|
| `data_root` | `data/energy_obj_tiles` | Корень датасета. Сплиты резолвятся как `data_root/<img_prefix>` и `data_root/<ann_file>`. |
| `log_root` | `log` | Корень логов. |
| `checkpoint_root` | `checkpoints` | Корень чекпойнтов. |
| `output_root` | `outputs` | Корень результатов оценки/инференса. |

### `[mmdet]` — выбор модели

| Параметр | Текущее | Дефолт | Описание |
|---|---|---|---|
| `config` | `mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py` | — | Базовый mmdet-конфиг. Префикс `mmdet::` — конфиг из установленного пакета mmdet. Переопределяется `--mmdet-config`. |
| `checkpoint` | `""` | `""` | Путь к весам для инициализации (`load_from`). Пусто — берутся init_cfg из базового конфига (для YOLOv3-d53 это `open-mmlab://darknet53`). Переопределяется `--checkpoint`. |
| `disable_pretrained` | `false` | `false` | `true` — обнулить pretrained-init backbone/neck (обучение с нуля). Для нашей задачи оставлено `false`: датасет маленький, COCO-pretrained обязателен (обоснование — [training_recommendations.md](training_recommendations.md), «Выбор конфигурации»). |

### `[dataset]`

| Параметр | Текущее | Дефолт | Описание |
|---|---|---|---|
| `type` | `YOLOTxtDataset` | — | Локальный датасет YOLO-разметки (`src/magicnet_2d/mmdet_datasets.py`): читает `<img_prefix>` + `.txt`-метки `class cx cy w h` (нормализованные). |
| `classes` | `substation`, `supports 0,4`, `supports 10`, `supports 35`, `transformer` | — | Имена классов; задаёт `num_classes` модели и metainfo. Порядок должен совпадать с id в метках. |
| `class_id_map` | не задан | `null` | Перекодировка исходных class id → индексы `classes`. Не нужна — id в метках идут 0..4 подряд. |
| `filter_empty_gt` | `false` | `null` | Только train-сплит: `false` — НЕ отбрасывать чипы без боксов. Критично: в train 12.8k фоновых чипов из 15.5k, базовый mmdet-YOLOv3 конфиг ставит `filter_empty_gt=True` и молча выбрасывает фон → взрыв ложных срабатываний на фоне. См. чеклист в [training_recommendations.md](training_recommendations.md). |

Сплиты (для каждого: `ann_file` — директория меток, `img_prefix` — директория изображений):

| Секция | `ann_file` | `img_prefix` | Чипов |
|---|---|---|---|
| `[dataset.train]` | `labels/train` | `images/train` | 7515 |
| `[dataset.val]` | `labels/val` | `images/val` | 1623 |
| `[dataset.test]` | `labels/test` | `images/test` | 1510 |

Сплиты наследуются от родительских кадров (см. `metadata.json` датасета); утечки train→val/test нет.

Важно соотношение фона: датасет пересобран 2026-11-16 с `--empty-frac 0.02`, фоновые чипы составляют **16.3% train** (1226 из 7515). Ранее при `--empty-frac 0.15` фон занимал 59% train — обучение YOLOv3 на таком составе дало коллапс objectness и нулевые метрики (см. таблицу «Типичные проблемы» и чеклист в [training_recommendations.md](training_recommendations.md)).

### `[train]`

| Параметр | Текущее | Дефолт (config.py) | Описание |
|---|---|---|---|
| `seed` | `42` | `42` | Сид для генераторов/torch. |
| `validate` | `true` | `true` | Включить валидацию. `false` — val_dataloader/val_cfg/val_evaluator обнуляются, обучение идёт без метрик. |
| `amp` | `false` | `false` | Mixed precision (`AmpOptimWrapper`). RTX 4090 выдерживает текущий batch без AMP; включение ускорит эпоху, но для YOLOv3 выигрыш невелик. |
| `max_epochs` | `100` | из базового конфига mmdet (273) | Длительность обучения. Переопределяет базовый `EpochBasedTrainLoop`. |
| `val_interval` | не задан | из базового конфига mmdet (**7**) | Каждые N эпох считать валидацию. У YOLOv3-coco база ставит 7; валидация на 3206 чипах с метрикой занимает заметное время, при желании частоту менять здесь. |
| `batch_size` | `8` | из базового конфига (8) | Batch train/val/test дataloader'ов. |
| `num_workers` | `8` | `null` (не трогать) | Воркеры дataloader'ов. `0`/`null` — отключает многопроцессную загрузку и `persistent_workers`. 8 воркеров даёт `data_time ≈ 0.003 c` (загрузка не боттleneck). |
| `checkpoint_interval` | `2` | `1` | Чекпойнт каждые N эпох. |
| `max_keep_checkpoints` | `3` | `3` | Хранить последние N чекпойнтов (старшие удаляются). ~235 МБ на чекпойнт YOLOv3-d53. |
| `logger_interval` | `50` | `50` | Период вывода train-лога, итераций. |
| `resume` | `""` | `null` | Путь к чекпойнту для возобновления (модель + оптимизатор + эпоха). Пусто — обучение с начала. Для «чистого» дообучения весов без состояния оптимизатора используйте `[mmdet] checkpoint`. |

### `[inference]` — параметры `scripts/infer.py`

| Параметр | Текущее | Дефолт | Описание |
|---|---|---|---|
| `device` | `auto` | `auto` | `auto`/`cuda`/`cpu`. |
| `score_threshold` | `0.3` | `0.3` | Порог уверенности; также передаётся в YOLOTxtMetric как `score_thr`. Диапазон [0, 1]. |
| `batch_size` | `1` | `1` | Batch инференса. |
| `output_subdir` | `inference` | `inference` | Поддиректория в `outputs/` для результатов. |

## Что НЕ управляется этим TOML (наследуется из базового mmdet-конфига)

Эти вещи жёстко приходят из `mmdet::yolo/yolov3_d53_8xb8-ms-608-273e_coco.py`; в TOML рычагов нет, меняются только заменой базового конфига или кастомизацией пайплайна:

- **Вход 608×608.** Train: `RandomResize` 320–608; val/test: `Resize(608)`. Тайлы 1280 px сжимаются в 608 — объекты 50–165 px становятся ~24–78 px. Это главный зарезервированный рычаг роста качества: поднять вход до 1280 (тайл = вход без ресайза, см. [training_recommendations.md](training_recommendations.md), «Как выбрать размер тайла»). Требует оверрайда пайплайна — отдельная задача.
- **Аугментации:** Expand, MinIoURandomCrop, RandomFlip, PhotoMetricDistortion (стандарт YOLOv3-coco).
- **`val_interval = 7`** — пока не задан явно в TOML (см. таблицу `[train]`).

## Типовые изменения

- **Сменить детектор** (например, на RT-DETR/Faster R-CNN по рекомендациям): поменять `mmdet.config` на нужный `mmdet::<path>.py`. Структура конфига (датасет, сплиты, train-параметры) не меняется. Проверить совместимость head'а с числом классов и `load_from`. Готовый пример — `configs/project/energy_obj_tiles_frcnn_mmdet.toml` (Faster R-CNN r50 FPN, COCO-pretrained веса лежат в `checkpoints/`, подключение через `mmdet.checkpoint`; RT-DETR в установленном mmdet 3.3.0 отсутствует).
- **Поменять длительность:** `train.max_epochs` (и при необходимости `val_interval`).
- **Продолжить прерванное обучение:** `train.resume = "checkpoints/energy_obj_tiles_mmdet/epoch_XX.pth"`, перезапуск той же команды. Логи допишутся в тот же ран.
- **Новый осмысленный прогон:** новый `run_name` (или `--run-name`), чтобы не смешивать артефакты.
- **Переключить AMP:** `train.amp = true`.
- **Пороги инференса:** `inference.score_threshold` (0.05–0.6 для анализа PR-кривой на val).

## Типичные проблемы и симптомы

| Симптом | Причина | Что делать |
|---|---|---|
| mAP около нуля даже после десятков эпох, ложных срабатываний много | `filter_empty_gt` стал `true`/пропал — фоновые чипы отфильтрованы, модель не видит фон | вернуть `filter_empty_gt = false`; проверить в `log/<run>/resolved_config.py` наличие `filter_cfg=dict(filter_empty_gt=False)` |
| mAP около нуля, TP=0, FP=0; сырые выходы: objectness ≈ 0, классы живые; loss_conf доминирует и почти не падает | коллапс objectness от перегруза фоном: доля пустых чипов в train ~50%+; диагностировано на ране `energy_obj_tiles_mmdet` (100 эпох, mAP 0.000) | пересобрать тайлы с меньшим `--empty-frac` (фон 10–20% train), перезапустить обучение с нуля; проверять `energy/mAP@0.5` уже с первой валидации |
| `CUDA out of memory` | batch/вход слишком большие | уменьшить `train.batch_size`; при смене детектора — вход модели |
| Валидация не запускается | `validate = false` или `val_interval` больше `max_epochs` | включить `validate`, выставить разумный `val_interval` |
| Метрики есть, но неизменно низкие, loss нормальный | порог `score_thr` метрики не подходит | смотреть FP/FN-дампы в `outputs/evaluation/<run>/`, подбирать `score_threshold` |
| Редкие классы (substation — 7 боксов) не учатся | дисбаланс классов | меры из [training_recommendations.md](training_recommendations.md), «Дополнительные меры» |

## Связанное

- `data/energy_obj_tiles/metadata.json` — параметры тайлинга и статистика датасета.
- `tests/test_config.py`, `tests/test_mmdet_support.py` — проверки загрузки конфига и сборки runtime-конфига (включая `filter_empty_gt`).
- `doc/training_recommendations.md` — чеклист внедрения и обоснование параметров.
