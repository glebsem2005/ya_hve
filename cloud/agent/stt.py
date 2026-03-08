"""Yandex SpeechKit STT — voice message to text.

Uses synchronous recognition API for short audio (< 30s).
Expects OGG Opus format (Telegram voice messages).
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


async def recognize_voice(audio_bytes: bytes) -> str | None:
    """Recognize speech from OGG Opus audio bytes.

    Returns recognized text or None on failure.
    """
    if not YANDEX_API_KEY:
        logger.error("YANDEX_API_KEY not set, cannot use STT")
        return None

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                STT_URL,
                params={
                    "folderId": YANDEX_FOLDER_ID,
                    "lang": "ru-RU",
                    "format": "oggopus",
                },
                headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
                content=audio_bytes,
            )

        if resp.status_code != 200:
            logger.error("SpeechKit STT error %s: %s", resp.status_code, resp.text)
            return None

        result = resp.json().get("result")
        if not result:
            logger.warning("SpeechKit returned empty result")
            return None

        return result

    except Exception as e:
        logger.error("SpeechKit STT failed: %s", e)
        return None
