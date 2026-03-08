"""RAG agent — File Search + Web Search via Yandex AI Studio.

Uses YandexGPT with tools (file_search, web_search) when SEARCH_INDEX_ID
is configured. Falls back to plain YandexGPT completion otherwise.

Environment variables:
  YANDEX_API_KEY      — API key for Yandex Cloud
  YANDEX_FOLDER_ID    — Yandex Cloud folder ID
  SEARCH_INDEX_ID     — File Search index ID (optional, enables RAG)
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
SEARCH_INDEX_ID = os.getenv("SEARCH_INDEX_ID")

API_URL = os.getenv(
    "YANDEX_GPT_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)

SYSTEM_PROMPT = """
Ты — юридический помощник системы мониторинга леса ForestGuard.
Твоя задача — помочь лесному инспектору:
1. Определить нарушение по Лесному кодексу РФ, КоАП, УК РФ
2. Составить протокол или рекомендацию по действиям
3. Указать конкретные статьи закона

Правила:
- Пиши по-русски, структурировано
- Ссылайся на конкретные статьи (ЛК РФ, КоАП, УК РФ)
- Давай пошаговые инструкции
- Если данных недостаточно — укажи что нужно уточнить
"""

CLASS_CONTEXT = {
    "chainsaw": "незаконная рубка леса (звук бензопилы)",
    "gunshot": "незаконная охота / браконьерство (звук выстрела)",
    "engine": "несанкционированный заезд техники в лес (звук двигателя)",
    "axe": "незаконная рубка леса (звук топора)",
    "fire": "лесной пожар (звук огня / треск)",
}


async def _call_yandex_with_tools(prompt: str) -> str:
    """Call YandexGPT with File Search and Web Search tools."""
    tools = []

    if SEARCH_INDEX_ID:
        tools.append(
            {
                "type": "file_search",
                "file_search": {
                    "search_index_ids": [SEARCH_INDEX_ID],
                    "max_num_results": 5,
                },
            }
        )

    tools.append({"type": "web_search", "web_search": {}})

    body = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
        "completionOptions": {
            "stream": False,
            "temperature": 0.2,
            "maxTokens": 500,
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": prompt},
        ],
    }

    if tools:
        body["tools"] = tools

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
            json=body,
        )

    if resp.status_code != 200:
        logger.error("YandexGPT RAG error %s: %s", resp.status_code, resp.text)
        return _fallback_response(prompt)

    return resp.json()["result"]["alternatives"][0]["message"]["text"]


async def _call_yandex_plain(prompt: str) -> str:
    """Fallback: plain YandexGPT without tools."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
            json={
                "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.2,
                    "maxTokens": 500,
                },
                "messages": [
                    {"role": "system", "text": SYSTEM_PROMPT},
                    {"role": "user", "text": prompt},
                ],
            },
        )

    if resp.status_code != 200:
        logger.error("YandexGPT plain error %s: %s", resp.status_code, resp.text)
        return _fallback_response(prompt)

    return resp.json()["result"]["alternatives"][0]["message"]["text"]


def _fallback_response(prompt: str) -> str:
    """Static fallback when API is unavailable."""
    return (
        "⚠️ YandexGPT временно недоступен.\n\n"
        "Базовые рекомендации:\n"
        "1. Зафиксируйте GPS-координаты\n"
        "2. Сделайте фото/видео нарушения\n"
        "3. Не вступайте в конфликт\n"
        "4. Вызовите патрульную группу\n"
        "5. Составьте акт по форме (ст. 96 ЛК РФ)"
    )


async def query_action(audio_class: str, lat: float, lon: float) -> str:
    """Get action recommendations for a detected event."""
    context = CLASS_CONTEXT.get(audio_class, f"неизвестное нарушение ({audio_class})")

    prompt = (
        f"Обнаружено: {context}\n"
        f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n\n"
        f"Что должен сделать лесной инспектор?\n"
        f"Дай пошаговую инструкцию с ссылками на статьи закона."
    )

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_tools(prompt)
    return await _call_yandex_plain(prompt)


async def query_protocol(audio_class: str, lat: float, lon: float) -> str:
    """Get protocol template for a detected event."""
    context = CLASS_CONTEXT.get(audio_class, f"неизвестное нарушение ({audio_class})")

    prompt = (
        f"Обнаружено: {context}\n"
        f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n\n"
        f"Составь шаблон протокола об административном правонарушении.\n"
        f"Укажи применимые статьи КоАП/УК РФ, необходимые данные для заполнения."
    )

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_tools(prompt)
    return await _call_yandex_plain(prompt)


async def query_rag(question: str, context: str = "") -> str:
    """General-purpose RAG query for the REST API endpoint."""
    prompt = question
    if context:
        prompt = f"Контекст: {context}\n\nВопрос: {question}"

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_tools(prompt)
    return await _call_yandex_plain(prompt)
