# COCO8 For Local YOLO Smoke Tests

## Что это

`COCO8` — это официальный мини-датасет от Ultralytics для быстрой проверки пайплайна object detection на YOLO. Внутри всего 8 изображений:

- 4 изображения в `train`;
- 4 изображения в `val`.

Этого достаточно, чтобы быстро проверить:

- что обучение запускается;
- что TensorBoard пишет события;
- что сохраняются веса;
- что инференс и сохранение предсказаний работают корректно.

## Источник

- Официальный архив: `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip`
- Локальный манифест загрузки: `.source.json`
- Официальная документация Ultralytics по COCO8: `https://docs.ultralytics.com/datasets/detect/coco8/`

В проект датасет был скачан и распакован в каталог:

- `data/yolo/coco8_ultralytics`

## Структура

```text
data/yolo/coco8_ultralytics/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── LICENSE
└── README.md
```

Структура разметки соответствует формату YOLO:

- для каждого изображения есть `.txt`-файл с bounding boxes;
- каждая строка хранит `class_id x_center y_center width height`;
- координаты нормализованы в диапазон `0..1`.

## Состав датасета

Разбиение:

- `train`: 4 изображения;
- `val`: 4 изображения.

Классы, которые реально встречаются в этих 8 изображениях:

- `person`
- `dog`
- `horse`
- `elephant`
- `zebra`
- `giraffe`
- `umbrella`
- `bowl`
- `orange`
- `broccoli`
- `potted plant`
- `vase`

При этом dataset config использует стандартную 80-классовую схему COCO, потому что label ids в YOLO-разметке соответствуют полной нумерации COCO.

## Какие файлы используются для обучения и валидации

Для обучения используются:

- `images/train/*.jpg`
- `labels/train/*.txt`

Для валидации используются:

- `images/val/*.jpg`
- `labels/val/*.txt`

Dataset YAML для YOLO расположен здесь:

- `configs/yolo/coco8_ultralytics.yaml`

Основной training config расположен здесь:

- `configs/yolo/coco8_smoke.yaml`

## Зачем он подходит для этого проекта

Этот датасет выбран потому что он:

- маленький и быстро запускается локально;
- уже размечен в формате YOLO;
- официально поддерживается Ultralytics;
- хорошо подходит именно для smoke-test проверки всего цикла `train -> TensorBoard -> checkpoint -> inference`.
