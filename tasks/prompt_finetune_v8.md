# Prompt: Fine-tune YAMNet v8

> Использовать в отдельной сессии Claude Code, если нужно дообучить модель.

## Задача

Дообучить YAMNet v8 head model с учётом того, что демо-файлы теперь из реальных датасетов (ESC-50, UrbanSound8K). Текущая v7 обучена на FEATURE_DIM=2048.

Проверь, что v8 корректно классифицирует:
- `demo/audio/chainsaw.wav` -> chainsaw
- `demo/audio/gunshot.wav` -> gunshot
- `demo/audio/engine.wav` -> engine
- `demo/audio/axe.wav` -> axe (door_wood_knock из ESC-50, тренировочные данные axe были из FSC22 Kaggle)
- `demo/audio/fire.wav` -> fire

## Контекст

- Ноутбук обучения: `docs/notebooks/01_data_and_mix.ipynb`
- Источники данных: ESC-50 (chainsaw/engine/fire), UrbanSound8K classID=6 (gunshot), FSC22 (axe)
- Backgrounds: ESC-50 (rain/wind/birds/crickets/water), FSC22 (silence), Kaggle birds
- Файл классификатора: `edge/audio/classifier.py`
- YAMNET_CLASS_MAP для base YAMNet fallback -- проверить, что door_wood_knock из ESC-50 попадает в "Chop"/"Wood"/"Thump" для маппинга в axe
- FEATURE_DIM=2048, leak-free evaluation

## Зачем

Демо-файлы заменены с синтетических на реальные записи. Hardcoded override убран из `cloud/interface/main.py`. Теперь классификация должна работать честно -- и fine-tuned модель, и base YAMNet fallback.
