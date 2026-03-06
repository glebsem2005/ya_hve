import httpx
import json
import os
from dataclasses import dataclass

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
# TODO: Yandex Foundation Models completions API
# Docs: https://yandex.cloud/docs/foundation-models/
API_URL = os.getenv(
    "YANDEX_GPT_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)

SYSTEM_PROMPT = """
Ты — система мониторинга леса ForestGuard.
Получаешь данные от акустических датчиков и дрона.
Твоя задача: написать чёткий алерт егерю.

Правила:
- Пиши по-русски, коротко и конкретно
- Укажи координаты, тип угрозы, рекомендацию
- Без воды и лишних слов
- 2-3 предложения максимум
"""


@dataclass
class Alert:
    text: str
    priority: str
    lat: float
    lon: float


async def compose_alert(
    audio_class: str,
    visual_description: str,
    lat: float,
    lon: float,
    confidence: float,
) -> Alert:

    priority_map = {
        "chainsaw": "ВЫСОКИЙ",
        "gunshot": "ВЫСОКИЙ",
        "fire": "ВЫСОКИЙ",
        "unknown": "СРЕДНИЙ",
    }
    priority = priority_map.get(audio_class, "СРЕДНИЙ")
    prompt = f"""
Данные с датчиков:
- Звук: {audio_class} (уверенность {confidence:.0%})
- Визуальный анализ дрона: {visual_description}
- Координаты: {lat:.4f}°N, {lon:.4f}°E
- Приоритет: {priority}

Напиши алерт егерю.
"""
    text = await _call_yandex(prompt)
    return Alert(text=text, priority=priority, lat=lat, lon=lon)


async def _call_yandex(user_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
            json={
                "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.2,
                    "maxTokens": 200,
                },
                "messages": [
                    {"role": "system", "text": SYSTEM_PROMPT},
                    {"role": "user", "text": user_prompt},
                ],
            },
        )
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"]
