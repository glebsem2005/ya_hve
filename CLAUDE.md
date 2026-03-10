# Faun — AI-система акустического мониторинга леса

## Контекст

Яндекс кейс-чемпионат Social Tech Lab (при ВШЭ). Технический трек, направление — экология.
Команда из 6 человек. Защита: 11 марта 2026, 18:00, офис Яндекса.

**Продукт:** сеть микрофонов определяет и локализует нарушения (незаконная рубка, браконьерство, техника) в реальном времени.

## Архитектура

3 Docker-сервиса (`docker compose`):
- **cloud** (:8000) — FastAPI дашборд (Leaflet), Telegram-бот, YandexGPT алерты, RAG-агент
- **edge** — YAMNet classifier, TDOA триангуляция, decision engine, drone (ArduPilot)
- **lora_gateway** (:9000) — LoRa mesh relay

### Yandex Cloud AI Studio (ключевой критерий)

| Сервис | Файл |
|---|---|
| YandexGPT — алерты, юридизация | `cloud/agent/decision.py`, `bot_handlers.py` |
| AI Studio Assistants API — RAG-агент | `cloud/agent/rag_agent.py` |
| File Search (RAG) — 9 нормативных документов | `cloud/agent/rag_agent.py` |
| Web Search — актуальные правовые нормы | `cloud/agent/rag_agent.py` |
| SpeechKit STT — голосовые сообщения | `cloud/agent/stt.py` |
| Gemma 3 27B — анализ фото с дрона | `cloud/vision/classifier.py` |
| Yandex Workflows — 12-шаговый pipeline | `cloud/workflows/pipeline.py` |
| Classification Agent — AI-верификация | `cloud/agent/classification_agent.py` |
| DataSphere — обучение YAMNet v7 | `cloud/agent/datasphere_client.py` |
| DataLens — аналитический дашборд | `cloud/analytics/datalens.py` |

## VPS-сервер

| Параметр | Значение |
|---|---|
| IP | `81.85.73.178` |
| SSH | `ssh root@81.85.73.178` |
| Код | `/var/www/ya_hve`, ветка `main` |
| Docker | docker compose v2 (2.34.0) |
| Сервисы | cloud (:8000), edge, lora_gateway (:9000) |
| ОС | Ubuntu 22.04, 1.9 GB RAM |
| Модель | YAMNet v7 загружена и работает |

## ML Pipeline

- **YAMNet v7**: FEATURE_DIM=2048 (prod-compatible, PCEN/temporal off), leak-free evaluation
- **TDOA v5**: 10 improvements (subpixel, PHAT-beta, median GCC, MAD, DEMON)
- **Confidence gating**: 3 уровня (alert/verify/log)
- **Ноутбуки**: `docs/notebooks/` (01_data_and_mix, 02_yamnet_test, 03_distance_estimation)

## Структура

```
/
├── CLAUDE.md
├── cloud/                 # Cloud-сервис
│   ├── agent/             # YandexGPT, RAG, STT, Classification
│   ├── analytics/         # DataLens
│   ├── db/                # SQLite/YDB
│   ├── integrations/      # ФГИС ЛК
│   ├── interface/         # FastAPI + Leaflet дашборд
│   ├── notify/            # Telegram-бот
│   ├── vision/            # Gemma 3 / YandexGPT Vision
│   └── workflows/         # Yandex Workflows pipeline
├── edge/                  # Edge-сервис (YAMNet, TDOA)
├── gateway/               # LoRa gateway
├── devices/               # ESP32 firmware
├── simulator/             # Mic stream, drone sim
├── demo/                  # Demo audio files
├── tests/                 # Pytest
├── graphs/                # Presentation graphs
├── docs/                  # ML research & legal
│   ├── notebooks/         # Jupyter ноутбуки (3 шт.)
│   ├── results/           # CSV с метриками
│   ├── graphs/            # PNG графики из PoC
│   └── legal/             # Нормативные документы для RAG (9 шт.)
└── workflows/             # CI/CD
```

## Инструкции

- Язык общения: русский
- Docker: `docker compose` (не docker-compose)
- Ветка для коммитов: `main`
- Модели (*.keras, *.h5, *.npz) в .gitignore — загружены на VPS
- Telegram-бот: `@ya_faun_bot`
- FOLDER_ID: `b1g5lqh1mqg84cabtejb`, SEARCH_INDEX_ID: `fvttk7bjvnm39qogtoep`
