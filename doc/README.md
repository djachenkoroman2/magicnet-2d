# Документация датасета `energy_obj`

Здесь собрана документация на подготовленный датасет детекции объектов электроэнергетики `data/energy_obj` и на процесс его разметки.

## Документы

- [dataset.md](dataset.md) — полное описание датасета: источники, структура, сплиты, классы, статистика, geo-привязка, как пересобрать.
- [annotation.md](annotation.md) — подробное описание разметки: формат аннотаций, семантика классов, пайплайн разметки в Label Studio, правила качества и особые случаи.
- [training_recommendations.md](training_recommendations.md) — рекомендации по выбору конфигурации обучения детектора: диагноз прошлого конфига, тайлинг, сравнение mmdet/ultralytics, меры против дисбаланса классов.
- [config_energy_obj_tiles.md](config_energy_obj_tiles.md) — полный справочник по конфигу обучения на тайлах `configs/project/energy_obj_tiles_mmdet.toml`: все секции и параметры, обоснования, типовые изменения и типичные проблемы.
- [experiments.md](experiments.md) — сводная таблица экспериментов по архитектурам детекции на `energy_obj_tiles`: результаты всех прогонов, AP по классам, выводы; пополняется по мере новых ранов.

## Краткая сводка

- **Задача:** детекция объектов ЛЭП и подстанций на ортотрансформированных снимках.
- **Формат:** YOLO (текстовые `.txt`-метки), 5 классов: `substation`, `supports 0,4`, `supports 10`, `supports 35`, `transformer`.
- **Объём:** 864 изображения (тайлы ~10000×10000 px), 4910 боксов; сплиты train/val/test = 619/123/122.
- **Источники:** Label Studio-экспорты `energy2` (211 размеченных изображений) и `data/raw_part2/extracted` (10 проектов, 1369 изображений, из них 653 размеченных).
- **Пул на доразметку:** `data/energy_obj_unlabeled` — 717 изображений без аннотаций, ждут разметки и обратного мержа.
- **Обучение:** `configs/project/energy_obj_tiles_mmdet.toml` (MMDetection, `YOLOTxtDataset`, тайловый датасет; справочник — [config_energy_obj_tiles.md](config_energy_obj_tiles.md)). Обучение на целых кадрах прекращено (см. диагноз в [training_recommendations.md](training_recommendations.md)).

Автогенерируемые описания рядом с данными: `data/energy_obj/README.md`, `data/energy_obj/metadata.json`, `data/energy_obj_unlabeled/README.md`.
