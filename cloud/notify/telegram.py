"""Telegram notifications — zone-based routing to rangers.

Alerts are sent ONLY to rangers whose zone covers the event coordinates.
If no rangers cover the location, the alert is logged but NOT sent.

Rate limiting: each chat_id receives at most one alert per COOLDOWN_SECONDS
to prevent notification spam from rapid detections.
"""

import os
import time
import logging

from telegram import Bot
from telegram.constants import ParseMode
from cloud.agent.decision import Alert
from cloud.db.rangers import get_rangers_for_location, Ranger

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Minimum seconds between alerts to the same chat_id
COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", 300))

# chat_id → timestamp of last sent alert
_last_sent: dict[int, float] = {}


def _is_rate_limited(chat_id: int) -> bool:
    """Check if this chat_id was alerted recently."""
    last = _last_sent.get(chat_id, 0.0)
    return (time.monotonic() - last) < COOLDOWN_SECONDS


def _mark_sent(chat_id: int) -> None:
    _last_sent[chat_id] = time.monotonic()


def _get_target_chat_ids(lat: float, lon: float) -> list[int]:
    """Get chat IDs of rangers responsible for this location.

    Returns ONLY rangers whose zone covers the coordinates.
    If no rangers match, returns empty list (no fallback spam).
    """
    rangers = get_rangers_for_location(lat, lon)
    return [r.chat_id for r in rangers]


async def send_pending(lat: float, lon: float, audio_class: str, reason: str) -> None:
    """Send initial alert to all rangers covering this location."""
    bot = Bot(token=BOT_TOKEN)
    maps_link = f"https://maps.yandex.ru/?pt={lon},{lat}&z=15"

    text = (
        f"*Обнаружена аномалия*\n\n"
        f"Звук: `{audio_class}`\n"
        f"[{lat:.4f}°N, {lon:.4f}°E]({maps_link})\n\n"
        f"Дрон вылетел для подтверждения..."
    )

    chat_ids = _get_target_chat_ids(lat, lon)
    if not chat_ids:
        logger.info(
            "No rangers cover %.4f°N %.4f°E — pending alert not sent", lat, lon
        )
        return

    for chat_id in chat_ids:
        if _is_rate_limited(chat_id):
            logger.info("Rate-limited: skipping pending alert for chat_id=%s", chat_id)
            continue
        _mark_sent(chat_id)
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.error("Failed to send pending alert to %s: %s", chat_id, e)


async def send_confirmed(alert: Alert, photo_bytes: bytes | None) -> None:
    """Send confirmed alert with photo to all rangers covering this location."""
    bot = Bot(token=BOT_TOKEN)
    maps_link = f"https://maps.yandex.ru/?pt={alert.lon},{alert.lat}&z=15"

    caption = (
        f"{alert.priority}\n\n"
        f"{alert.text}\n\n"
        f"[{alert.lat:.4f}°N, {alert.lon:.4f}°E]({maps_link})"
    )

    chat_ids = _get_target_chat_ids(alert.lat, alert.lon)
    if not chat_ids:
        logger.info(
            "No rangers cover %.4f°N %.4f°E — confirmed alert not sent",
            alert.lat, alert.lon,
        )
        return

    for chat_id in chat_ids:
        if _is_rate_limited(chat_id):
            logger.info("Rate-limited: skipping confirmed alert for chat_id=%s", chat_id)
            continue
        _mark_sent(chat_id)
        try:
            if photo_bytes:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_bytes,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=caption,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True,
                )
        except Exception as e:
            logger.error("Failed to send confirmed alert to %s: %s", chat_id, e)
