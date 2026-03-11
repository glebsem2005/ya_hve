"""Drone Bot — receives photos and triggers the vision → alert pipeline.

On demo day, photos are sent from a phone to simulate drone camera.
When a threat is detected, an incident is created and rangers are alerted
via the Ranger bot (@ya_faun_bot).

Commands:
  /start  - Welcome message

Message handlers:
  PHOTO   - Classify via Vision, trigger full pipeline if threat detected
"""

import base64
import logging
import random
import re

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)

_MD_ESCAPE_RE = re.compile(r"([_*\[\]`~>#+\-=|{}.!\\])")


def escape_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters in text."""
    return _MD_ESCAPE_RE.sub(r"\\\1", text)


try:
    from cloud.interface.main import broadcast
except ImportError:

    async def broadcast(_msg):
        pass


async def drone_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message."""
    await update.message.reply_text(
        "Faun Drone Bot\n\n"
        "Отправьте фото для анализа.\n"
        "При обнаружении угрозы будет создан инцидент "
        "и отправлен алерт инспекторам."
    )


async def drone_photo_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle photo — classify via Vision and trigger alert pipeline."""
    chat_id = update.effective_chat.id
    await update.message.reply_text("Анализирую фото...")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = bytes(await photo_file.download_as_bytearray())
        photo_b64 = base64.b64encode(photo_bytes).decode()

        from cloud.vision.classifier import classify_photo

        result = await classify_photo(photo_b64)

        # Broadcast vision result to web dashboard
        await broadcast(
            {
                "event": "vision_classified",
                "description": result.description,
                "has_felling": result.has_felling,
                "has_human": result.has_human,
                "has_fire": result.has_fire,
                "has_machinery": result.has_machinery,
                "is_threat": result.is_threat,
            }
        )
        await broadcast({"event": "drone_photo", "drone_b64": photo_b64})

        has_threat = result.is_threat

        if has_threat:
            # Map vision flags to audio class for pipeline compatibility
            if result.has_machinery:
                audio_class = "engine"
            elif result.has_felling:
                audio_class = "chainsaw"
            elif result.has_fire:
                audio_class = "fire"
            elif result.has_human:
                audio_class = "axe"
            else:
                audio_class = "engine"

            # Random coordinates in Varnavino forestry district
            lat = round(random.uniform(57.05, 57.55), 6)
            lon = round(random.uniform(44.60, 45.40), 6)
            confidence = 0.85

            # Step 1: Create incident + alert rangers via Ranger bot
            incident = None
            try:
                from cloud.notify.telegram import send_pending

                incident = await send_pending(
                    lat=lat,
                    lon=lon,
                    audio_class=audio_class,
                    reason=result.description,
                    confidence=confidence,
                    is_demo=True,
                    broadcast=True,
                )
            except Exception:
                logger.exception("Step 1 failed: send_pending")

            # Step 2: Compose alert text via YandexGPT (skip if no incident)
            alert = None
            if incident:
                try:
                    from cloud.agent.decision import compose_alert

                    alert = await compose_alert(
                        audio_class=audio_class,
                        visual_description=result.description,
                        lat=lat,
                        lon=lon,
                        confidence=confidence,
                    )
                except Exception:
                    logger.exception("Step 2 failed: compose_alert")

            # Step 3: Store drone photo in incident (sent after ranger accepts)
            if alert and incident:
                try:
                    from cloud.notify.telegram import send_confirmed

                    await send_confirmed(alert, photo_bytes, incident)
                except Exception:
                    logger.exception("Step 3 failed: send_confirmed")

            try:
                await broadcast(
                    {
                        "event": "alert_sent",
                        "audio_class": audio_class,
                        "lat": lat,
                        "lon": lon,
                    }
                )
                await broadcast({"event": "pipeline_end", "reason": "incident_created"})
            except Exception:
                logger.exception("Broadcast failed after threat pipeline")

            desc_escaped = escape_markdown(result.description)
            reply = (
                f"*Обнаружена угроза*\n\n"
                f"Класс: {audio_class}\n"
                f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n"
                f"Визуальный анализ: {desc_escaped}\n"
                f"{'Инцидент создан, алерт отправлен.' if incident else 'Не удалось создать инцидент.'}"
            )
            try:
                await update.message.reply_text(reply, parse_mode="Markdown")
            except BadRequest:
                logger.warning(
                    "Markdown parse failed in threat reply, falling back to plain text"
                )
                await update.message.reply_text(reply)
        else:
            await broadcast({"event": "pipeline_end", "reason": "no_threat"})

            desc_escaped = escape_markdown(result.description)
            reply = f"*Анализ фото:*\n{desc_escaped}\n\nНарушений не обнаружено."
            try:
                await update.message.reply_text(reply, parse_mode="Markdown")
            except BadRequest:
                logger.warning(
                    "Markdown parse failed in no-threat reply, falling back to plain text"
                )
                await update.message.reply_text(reply)

    except Exception:
        logger.exception("Drone photo classification failed")
        await update.message.reply_text(
            "Не удалось проанализировать фото. Попробуйте ещё раз."
        )


def get_drone_handlers() -> list:
    """Return all handlers for the Drone bot."""
    return [
        CommandHandler("start", drone_start),
        MessageHandler(filters.PHOTO, drone_photo_handler),
    ]
