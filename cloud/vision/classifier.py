import json
import logging
import httpx
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

GEMMA_URL = "https://ai.api.cloud.yandex.net/v1/chat/completions"

VISION_URL = os.getenv(
    "YANDEX_VISION_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)

PROMPT = """Проанализируй снимок с дрона в лесу.
Ответь только JSON без markdown, формат:
{
  "description": "краткое описание что на снимке (1-2 предложения)",
  "has_human": true/false,
  "has_fire": true/false,
  "has_felling": true/false,
  "has_machinery": true/false,
  "is_threat": true/false
}
has_machinery = true если видна тяжёлая техника (трактор, экскаватор, лесовоз, харвестер и т.п.).
is_threat = true если видны признаки нарушения: незаконная рубка, браконьерство, поджог, подозрительная техника, человек с инструментом (топор, бензопила) в лесу. Туристы, грибники, мероприятия, дороги — это НЕ угроза."""


@dataclass
class VisionResult:
    description: str
    has_human: bool
    has_fire: bool
    has_felling: bool
    has_machinery: bool
    is_threat: bool


def _parse_result(raw: str) -> VisionResult:
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vision: failed to parse JSON: %s", raw[:200])
        return _stub_result()
    has_human = data.get("has_human", False)
    has_felling = data.get("has_felling", False)
    has_fire = data.get("has_fire", False)
    is_threat = data.get("is_threat", False)

    # Safety net: override LLM when obvious threat signals present
    has_machinery = data.get("has_machinery", False)
    description = data.get("description") or ""
    _TOOL_KEYWORDS = ("топор", "бензопил", "пила", "ружь", "винтовк", "оружи")
    desc_has_tool = any(kw in description.lower() for kw in _TOOL_KEYWORDS)
    if has_machinery:
        is_threat = True
    if has_human and (has_felling or has_fire or desc_has_tool):
        is_threat = True

    return VisionResult(
        description=description,
        has_human=has_human,
        has_fire=has_fire,
        has_felling=has_felling,
        has_machinery=has_machinery,
        is_threat=is_threat,
    )


async def _try_gemma(client: httpx.AsyncClient, photo_b64: str) -> VisionResult:
    """Gemma 3 27B — multimodal, OpenAI-compatible API."""
    resp = await client.post(
        GEMMA_URL,
        headers={
            "Authorization": f"Bearer {YANDEX_API_KEY}",
            "OpenAI-Project": YANDEX_FOLDER_ID,
        },
        json={
            "model": f"gpt://{YANDEX_FOLDER_ID}/gemma-3-27b-it/latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        },
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _parse_result(raw)


async def _try_yandexgpt_vision(
    client: httpx.AsyncClient, photo_b64: str
) -> VisionResult:
    """yandexgpt-vision-lite — Foundation Models API."""
    resp = await client.post(
        VISION_URL,
        headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
        json={
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-vision-lite",
            "completionOptions": {"temperature": 0.1},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        },
    )
    resp.raise_for_status()
    raw = resp.json()["result"]["alternatives"][0]["message"]["text"]
    return _parse_result(raw)


def _stub_result() -> VisionResult:
    """Conservative fallback: flag as potential threat so pipeline continues."""
    return VisionResult(
        description="Визуальный анализ недоступен. Требуется ручная проверка.",
        has_human=False,
        has_fire=False,
        has_felling=True,
        has_machinery=False,
        is_threat=True,
    )


async def classify_photo(photo_b64: str) -> VisionResult:
    """Classify a drone photo: Gemma 3 → yandexgpt-vision-lite → stub."""
    async with httpx.AsyncClient(timeout=45) as client:
        # 1. Gemma 3 27B (multimodal, AI Studio)
        try:
            result = await _try_gemma(client, photo_b64)
            logger.info("Vision: Gemma 3 27B OK")
            return result
        except Exception as e:
            logger.warning("Vision: Gemma 3 failed: %s", e)

        # 2. Fallback: yandexgpt-vision-lite
        try:
            result = await _try_yandexgpt_vision(client, photo_b64)
            logger.info("Vision: yandexgpt-vision-lite OK")
            return result
        except Exception as e:
            logger.warning("Vision: yandexgpt-vision-lite failed: %s", e)

    # 3. Fallback: realistic stub
    logger.warning("Vision: all models failed, using stub")
    return _stub_result()
