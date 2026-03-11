# Lessons Learned

## 1. Vision stub не должен быть "безопасным"

False negative хуже false positive для систем безопасности. Если Vision API
недоступен, stub должен сигнализировать потенциальную угрозу (`has_felling=True`),
чтобы pipeline продолжил работу и инспектор мог проверить вручную.

## 2. sync YDB из async = блокировка event loop

`pool.retry_operation_sync()` — синхронный вызов. Вызов из async-хандлера Telegram
блокирует event loop. При YDB rate limiting (`ResourceExhausted`) ретраи с
`time.sleep()` полностью замораживают обработку сообщений.

**Правило:** всегда `await asyncio.to_thread(create_incident, ...)` для YDB
операций из async-контекста.

## 3. YDB Serverless rate limits

Бесплатный тарифный план YDB Serverless имеет жёсткие RU-лимиты. Bulk upserts
15 batch по 200 строк исчерпывают квоту начиная с Batch 2.

**Правила:**
- `batch_size = 500` (меньше gRPC вызовов)
- `time.sleep(1.0)` между батчами (не 0.5)
- `wait = 2 ** (attempt + 1)` для retry (начинать с 2s, не 1s)
- DDL операции тоже throttle: `time.sleep(0.5)` между ними

## 4. Тестировать деградацию

Если Vision API упал, pipeline должен продолжать работать. Нельзя полагаться на
внешние API без graceful degradation. JSON от Vision может быть битым — `json.loads`
обязательно в `try/except`.

## 5. `session.prepare()` vs typed tuples

`session.prepare()` парсит DECLARE и сам типизирует параметры. Typed tuples
(`TypedValue(PrimitiveType.Utf8, value)`) нужны ТОЛЬКО без `prepare()`.
Не надо дублировать типизацию — это вызывает `double type binding` ошибки.

## 6. Module-level sync код блокирует import

`seed_microphones()` на уровне модуля (вне `lifespan`) вызывается синхронно при
`import main`. Если YDB недоступен или медленный — весь FastAPI зависает на старте.

**Правило:** тяжёлые sync-операции перенести в `lifespan` через `asyncio.to_thread()`.

## 7. Docker volume mount + restart ≠ свежий код

При `docker compose restart` контейнер НЕ пересоздаётся — может закэшировать старый
`.pyc` или состояние модулей. Для деплоя с volume mount:
- `docker compose up -d --force-recreate` (пересоздать контейнер)
- Или `docker compose up --build -d` (пересобрать образ + пересоздать)
- Добавить `ENV PYTHONDONTWRITEBYTECODE=1` в Dockerfile чтобы Python не писал `.pyc`

## 8. При рефакторинге — проверять все использования удалённых переменных

Удалили `mic_coords = [...]` при рефакторинге `_run_demo()`, но `MicSimulator`
использовал `mic_coords` ниже по коду. `grep mic_coords` перед удалением спас бы
от NameError на проде.

## 9. OOM Kill от дублирования TF в контейнерах

Cloud-контейнер импортирует `from edge.audio.classifier import classify` напрямую
(Python import, не HTTP). Это загружает TensorFlow (~500-800 MB) повторно —
edge уже держит свой экземпляр TF. На VPS с 1.9 GB RAM два TF = OOM Kill
(exitCode=137, SIGKILL).

**Правила:**
- Auto-demo (`_auto_demo()`) триггерит TF-загрузку; на малой RAM → `DISABLE_AUTO_DEMO=1`
- Healthcheck: `curl -sf` вместо `python -c "..."` (экономия ~40 MB на проверку)
- `start_period: 360s` для healthcheck (TF грузится ~6 минут)
- Долгосрочно: cloud → edge по HTTP, не через Python import

## 10. Healthcheck не должен быть тяжелее самого сервиса

`python -c "import urllib.request; ..."` запускает полный Python-интерпретатор
(~40 MB) для каждой проверки. На VPS с ограниченной RAM это усиливает memory
pressure во время загрузки TF. `curl -sf url` использует ~5 MB и работает мгновенно.
