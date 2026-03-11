"""Telegram notifications — zone-based routing to rangers.

Alerts are sent ONLY to rangers whose zone covers the event coordinates.
If no rangers cover the location, the alert is logged but NOT sent.

Rate limiting: each chat_id receives at most one alert per COOLDOWN_SECONDS
to prevent notification spam from rapid detections.
"""

import asyncio
import io
import os
import time
import logging
from datetime import datetime, timezone, timedelta

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from cloud.agent.decision import Alert
from cloud.db.rangers import (
    get_rangers_for_location,
    get_all_rangers,
    get_nearest_rangers,
    Ranger,
)
from cloud.db.rangers import _haversine as haversine_m
from cloud.db.incidents import Incident, create_incident, get_recent_nearby_incident

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

MOSCOW_TZ = timezone(timedelta(hours=3))

# Minimum seconds between alerts to the same chat_id (default fallback)
COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", 300))

# Severity-aware cooldowns (seconds)
SEVERITY_COOLDOWNS: dict[str, int] = {
    "alert": 60,
    "verify": 300,
    "log": 600,
}

# Quiet hours (Moscow time)
QUIET_HOURS_START = int(os.getenv("QUIET_HOURS_START", 22))
QUIET_HOURS_END = int(os.getenv("QUIET_HOURS_END", 6))

# chat_id → timestamp of last sent alert
_last_sent: dict[int, float] = {}

CLASS_NAME_RU: dict[str, str] = {
    "chainsaw": "Бензопила",
    "gunshot": "Выстрел",
    "engine": "Двигатель",
    "axe": "Топор",
    "fire": "Огонь",
    "background": "Фон",
    "unknown": "Неизвестно",
}

GATING_LABEL: dict[str, str] = {
    "alert": "ТРЕВОГА",
    "verify": "ПРОВЕРКА",
    "log": "ЖУРНАЛ",
}

PRIORITY_LABEL: dict[str, str] = {
    "ВЫСОКИЙ": "Приоритет: ВЫСОКИЙ",
    "СРЕДНИЙ": "Приоритет: СРЕДНИЙ",
    "НИЗКИЙ": "Приоритет: НИЗКИЙ",
}


def _is_rate_limited(chat_id: int, gating_level: str = "verify") -> bool:
    """Check if this chat_id was alerted recently (severity-aware)."""
    last = _last_sent.get(chat_id, 0.0)
    cooldown = SEVERITY_COOLDOWNS.get(gating_level, COOLDOWN_SECONDS)
    return (time.monotonic() - last) < cooldown


def _is_quiet_hours(gating_level: str = "verify") -> bool:
    """Check if current Moscow time is within quiet hours.

    Critical alerts (gating_level='alert') bypass quiet hours.
    """
    if gating_level == "alert":
        return False
    hour = datetime.now(tz=MOSCOW_TZ).hour
    if QUIET_HOURS_START > QUIET_HOURS_END:
        return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END
    return QUIET_HOURS_START <= hour < QUIET_HOURS_END


def _mark_sent(chat_id: int) -> None:
    _last_sent[chat_id] = time.monotonic()


def _get_target_chat_ids(lat: float, lon: float) -> list[int]:
    """Get chat IDs of active rangers whose zone covers (lat, lon)."""
    rangers = get_rangers_for_location(lat, lon)
    return [r.chat_id for r in rangers]


def _gating_level(confidence: float) -> str:
    if confidence >= 0.70:
        return "alert"
    if confidence >= 0.40:
        return "verify"
    return "log"


async def send_pending(
    lat: float,
    lon: float,
    audio_class: str,
    reason: str,
    confidence: float = 0.0,
    gating_level: str | None = None,
    is_demo: bool = False,
    broadcast: bool = False,
) -> Incident | None:
    """Send initial alert to all rangers covering this location.

    Creates an Incident and returns it. Drone photo is NOT sent yet —
    it will be sent after the ranger accepts the call.
    """
    level = gating_level or _gating_level(confidence)

    # Spatial deduplication: skip if a recent nearby incident exists
    existing = get_recent_nearby_incident(lat, lon, radius_m=500, max_age_s=300)
    if existing:
        logger.info(
            "Dedup: incident %s already covers %.4f,%.4f — skipping",
            existing.id,
            lat,
            lon,
        )
        return existing

    # Quiet hours: suppress non-critical alerts
    if _is_quiet_hours(level):
        logger.info(
            "Quiet hours: suppressing %s-level alert at %.4f,%.4f", level, lat, lon
        )
        return None

    bot = Bot(token=BOT_TOKEN)
    maps_url = f"https://maps.yandex.ru/?pt={lon},{lat}&z=15"

    class_ru = CLASS_NAME_RU.get(audio_class, audio_class)
    level_label = GATING_LABEL.get(level, level)
    conf_pct = f"{confidence:.0%}" if confidence else "---"
    timestamp = datetime.now(tz=MOSCOW_TZ).strftime("%H:%M:%S")

    incident = await asyncio.to_thread(
        create_incident,
        audio_class=audio_class,
        lat=lat,
        lon=lon,
        confidence=confidence,
        gating_level=level,
        is_demo=is_demo,
    )

    text = (
        f"*АЛЕРТ: {class_ru}*\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Время: {timestamp} МСК\n"
        f"Координаты: {lat:.4f} N, {lon:.4f} E\n"
        f"Уверенность: {conf_pct}\n"
        f"Уровень: {level_label}\n\n"
    )

    # Add nearest ranger info
    nearest = get_nearest_rangers(lat, lon, limit=1)
    if nearest:
        dist_km = (
            haversine_m(lat, lon, nearest[0].current_lat, nearest[0].current_lon) / 1000
        )
        text += f"Ближайший инспектор: {nearest[0].name} ({dist_km:.1f} км)\n\n"

    keyboard_rows = [
        [InlineKeyboardButton("На карте", url=maps_url)],
        [
            InlineKeyboardButton(
                "Принять вызов",
                callback_data=f"accept:{incident.id}",
            ),
        ],
        [
            InlineKeyboardButton(
                "Отложить 15 мин",
                callback_data=f"snooze:{incident.id}",
            ),
        ],
    ]

    if level == "verify":
        text += "Требуется проверка (дрон не отправлен)"
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "Отправить дрона",
                    callback_data=f"dispatch_drone:{incident.id}",
                ),
            ]
        )
    else:
        text += "Дрон вылетел для подтверждения"

    keyboard = InlineKeyboardMarkup(keyboard_rows)

    if broadcast:
        rangers = get_all_rangers()
        chat_ids = [r.chat_id for r in rangers if r.active]
    else:
        chat_ids = _get_target_chat_ids(lat, lon)
    if not chat_ids:
        logger.info("No rangers cover %.4f N %.4f E — pending alert not sent", lat, lon)
        return incident

    for chat_id in chat_ids:
        if _is_rate_limited(chat_id, level):
            logger.info("Rate-limited: skipping pending alert for chat_id=%s", chat_id)
            continue
        _mark_sent(chat_id)
        try:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )
            incident.alert_message_ids[chat_id] = msg.message_id
        except Exception as e:
            logger.error("Failed to send pending alert to %s: %s", chat_id, e)

    return incident


async def send_pending_to_chat(
    chat_id: int,
    lat: float,
    lon: float,
    audio_class: str,
    reason: str,
    confidence: float = 0.0,
    gating_level: str | None = None,
    is_demo: bool = False,
) -> Incident | None:
    """Send alert directly to a specific chat_id (for /test command)."""
    bot = Bot(token=BOT_TOKEN)
    maps_url = f"https://maps.yandex.ru/?pt={lon},{lat}&z=15"

    class_ru = CLASS_NAME_RU.get(audio_class, audio_class)
    level = gating_level or _gating_level(confidence)
    level_label = GATING_LABEL.get(level, level)
    conf_pct = f"{confidence:.0%}" if confidence else "---"
    timestamp = datetime.now(tz=MOSCOW_TZ).strftime("%H:%M:%S")

    incident = await asyncio.to_thread(
        create_incident,
        audio_class=audio_class,
        lat=lat,
        lon=lon,
        confidence=confidence,
        gating_level=level,
        is_demo=is_demo,
    )

    text = (
        f"*АЛЕРТ: {class_ru}*\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Время: {timestamp} МСК\n"
        f"Координаты: {lat:.4f} N, {lon:.4f} E\n"
        f"Уверенность: {conf_pct}\n"
        f"Уровень: {level_label}\n\n"
    )

    keyboard_rows = [
        [InlineKeyboardButton("На карте", url=maps_url)],
        [
            InlineKeyboardButton(
                "Принять вызов",
                callback_data=f"accept:{incident.id}",
            ),
        ],
        [
            InlineKeyboardButton(
                "Отложить 15 мин",
                callback_data=f"snooze:{incident.id}",
            ),
        ],
    ]

    if level == "verify":
        text += "Требуется проверка (дрон не отправлен)"
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "Отправить дрона",
                    callback_data=f"dispatch_drone:{incident.id}",
                ),
            ]
        )
    else:
        text += "Дрон вылетел для подтверждения"

    keyboard = InlineKeyboardMarkup(keyboard_rows)

    try:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
            disable_web_page_preview=True,
        )
        incident.alert_message_ids[chat_id] = msg.message_id
    except Exception as e:
        logger.error("Failed to send test alert to %s: %s", chat_id, e)

    return incident


async def send_confirmed(
    alert: Alert, photo_bytes: bytes | None, incident: Incident | None = None
) -> None:
    """Send confirmed alert with photo to all rangers covering this location.

    If incident is provided, stores drone photo in the incident
    instead of sending directly (photo will be sent after accept).
    """
    if incident:
        # Store drone data in incident for later delivery
        if photo_bytes:
            import base64

            incident.drone_photo_b64 = base64.b64encode(photo_bytes).decode()
        incident.drone_comment = alert.text
        # Persist to DB (critical for YDB backend)
        from cloud.db.incidents import update_incident

        update_incident(
            incident.id,
            drone_photo_b64=incident.drone_photo_b64,
            drone_comment=incident.drone_comment,
        )
        return

    # Fallback: direct send (legacy behavior without incident workflow)
    bot = Bot(token=BOT_TOKEN)
    maps_url = f"https://maps.yandex.ru/?pt={alert.lon},{alert.lat}&z=15"
    priority_text = PRIORITY_LABEL.get(alert.priority, alert.priority)

    caption = (
        f"{priority_text}\n\n"
        f"{alert.text}\n\n"
        f"[{alert.lat:.4f} N, {alert.lon:.4f} E]({maps_url})"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("На карте", url=maps_url)],
        ]
    )

    chat_ids = _get_target_chat_ids(alert.lat, alert.lon)
    if not chat_ids:
        logger.info(
            "No rangers cover %.4f N %.4f E — confirmed alert not sent",
            alert.lat,
            alert.lon,
        )
        return

    for chat_id in chat_ids:
        if _is_rate_limited(chat_id, "alert"):
            logger.info(
                "Rate-limited: skipping confirmed alert for chat_id=%s", chat_id
            )
            continue
        _mark_sent(chat_id)
        try:
            if photo_bytes:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_bytes,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=keyboard,
                )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=caption,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=keyboard,
                    disable_web_page_preview=True,
                )
        except Exception as e:
            logger.error("Failed to send confirmed alert to %s: %s", chat_id, e)


async def send_drone_photo(chat_id: int, incident: Incident) -> None:
    """Send drone photo with AI comment after ranger accepts the call."""
    bot = Bot(token=BOT_TOKEN)

    if incident.drone_photo_b64:
        import base64

        photo_bytes = base64.b64decode(incident.drone_photo_b64)
        caption = "Снимок с дрона"
        if incident.drone_comment:
            caption += f"\n\n{incident.drone_comment}"

        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=photo_bytes,
                caption=caption[:1024],
            )
        except Exception as e:
            logger.error("Failed to send drone photo to %s: %s", chat_id, e)
    elif incident.drone_comment:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"Комментарий системы:\n{incident.drone_comment}",
            )
        except Exception as e:
            logger.error("Failed to send drone comment to %s: %s", chat_id, e)


async def send_arrival_question(chat_id: int, incident: Incident) -> None:
    """Ask ranger what they found on site."""
    bot = Bot(token=BOT_TOKEN)

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Нарушение подтверждено",
                    callback_data=f"verdict:confirmed:{incident.id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Ложный вызов",
                    callback_data=f"verdict:false:{incident.id}",
                ),
            ],
        ]
    )

    try:
        await bot.send_message(
            chat_id=chat_id,
            text="Вы рядом с точкой. Что на месте?",
            reply_markup=keyboard,
        )
    except Exception as e:
        logger.error("Failed to send arrival question to %s: %s", chat_id, e)


async def send_evidence_request(chat_id: int) -> None:
    """Ask ranger to send photo and description."""
    bot = Bot(token=BOT_TOKEN)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text="Пришлите фото нарушения и опишите ситуацию (текстом или голосовым сообщением).",
        )
    except Exception as e:
        logger.error("Failed to send evidence request to %s: %s", chat_id, e)


async def send_protocol_pdf(chat_id: int, pdf_bytes: bytes) -> None:
    """Send generated PDF protocol to ranger."""
    bot = Bot(token=BOT_TOKEN)
    try:
        await bot.send_document(
            chat_id=chat_id,
            document=io.BytesIO(pdf_bytes),
            filename="protocol.pdf",
            caption="Протокол сформирован.",
        )
    except Exception as e:
        logger.error("Failed to send protocol PDF to %s: %s", chat_id, e)
