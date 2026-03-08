"""Telegram bot handlers for ranger self-registration and alerts.

Commands:
  /start  - Begin registration or show welcome if already registered
  /status - Show current registration details
  /stop   - Deactivate alerts (ranger remains in DB but active=False)
  /test   - Send a test alert with inline buttons for demo
"""

import base64
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from cloud.db.rangers import add_ranger, get_ranger_by_chat_id, set_active
from cloud.notify.districts import DISTRICTS
from cloud.notify.telegram import send_pending_to_chat, CLASS_NAME_RU

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — register new ranger or greet existing one."""
    chat_id = update.effective_chat.id
    existing = get_ranger_by_chat_id(chat_id)

    if existing:
        if not existing.active:
            set_active(chat_id, True)
            await update.message.reply_text(
                f"С возвращением, {existing.name}! Оповещения снова включены."
            )
        else:
            await update.message.reply_text(
                f"Вы уже зарегистрированы, {existing.name}.\n"
                "Используйте /status для проверки или /stop для отключения."
            )
        return

    keyboard = [
        [InlineKeyboardButton(d.name_ru, callback_data=f"district:{slug}")]
        for slug, d in DISTRICTS.items()
    ]
    await update.message.reply_text(
        "Добро пожаловать в ForestGuard!\n\nВыберите ваше лесничество:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def district_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button press — register ranger for chosen district."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("district:"):
        return

    slug = data.split(":", 1)[1]
    district = DISTRICTS.get(slug)
    if not district:
        await query.edit_message_text("Ошибка: неизвестное лесничество.")
        return

    chat_id = query.message.chat_id
    user = query.from_user
    name = user.full_name or user.username or str(chat_id)

    if get_ranger_by_chat_id(chat_id):
        await query.edit_message_text("Вы уже зарегистрированы! /status")
        return

    try:
        add_ranger(
            name=name,
            chat_id=chat_id,
            zone_lat_min=district.lat_min,
            zone_lat_max=district.lat_max,
            zone_lon_min=district.lon_min,
            zone_lon_max=district.lon_max,
        )
    except Exception:
        logger.exception("Failed to register ranger chat_id=%s", chat_id)
        await query.edit_message_text("Ошибка регистрации. Попробуйте позже.")
        return

    await query.edit_message_text(
        f"Вы зарегистрированы!\n\n"
        f"Лесничество: {district.name_ru}\n"
        f"Регион: {district.region_ru}\n\n"
        "Вы будете получать оповещения о подозрительной активности "
        "в вашей зоне. Используйте /stop для отключения."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show registration details."""
    chat_id = update.effective_chat.id
    ranger = get_ranger_by_chat_id(chat_id)

    if not ranger:
        await update.message.reply_text(
            "Вы не зарегистрированы. Отправьте /start для регистрации."
        )
        return

    state = "включены" if ranger.active else "отключены"
    await update.message.reply_text(
        f"Имя: {ranger.name}\n"
        f"Зона: {ranger.zone_lat_min:.2f}–{ranger.zone_lat_max:.2f}°N, "
        f"{ranger.zone_lon_min:.2f}–{ranger.zone_lon_max:.2f}°E\n"
        f"Оповещения: {state}"
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop — deactivate alerts."""
    chat_id = update.effective_chat.id
    ranger = get_ranger_by_chat_id(chat_id)

    if not ranger:
        await update.message.reply_text("Вы не зарегистрированы.")
        return
    if not ranger.active:
        await update.message.reply_text("Оповещения уже отключены.")
        return

    set_active(chat_id, False)
    await update.message.reply_text(
        "Оповещения отключены. Отправьте /start чтобы включить снова."
    )


async def test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /test — send a demo alert to current chat for testing buttons."""
    chat_id = update.effective_chat.id
    await update.message.reply_text("🧪 Отправляю тестовый алерт...")
    await send_pending_to_chat(
        chat_id=chat_id,
        lat=55.7512,
        lon=37.6135,
        audio_class="chainsaw",
        reason="Test alert",
        confidence=0.92,
        gating_level="alert",
    )


async def handle_inspector_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle photo from inspector — classify via YandexGPT Vision."""
    await update.message.reply_text("📷 Анализирую фото...")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        photo_b64 = base64.b64encode(photo_bytes).decode()

        from cloud.vision.classifier import classify_photo

        result = await classify_photo(photo_b64)

        reply = f"📷 *Анализ фото:*\n{result.description}\n\n"
        if result.has_felling:
            reply += "🪓 Обнаружена рубка\n"
        if result.has_human:
            reply += "👤 Обнаружены люди\n"
        if result.has_fire:
            reply += "🔥 Обнаружен огонь\n"
        if not (result.has_felling or result.has_human or result.has_fire):
            reply += "✅ Нарушений не обнаружено\n"

        await update.message.reply_text(reply, parse_mode="Markdown")
    except Exception as e:
        logger.exception("Photo classification failed")
        await update.message.reply_text(f"❌ Ошибка анализа фото: {e}")


async def rag_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle RAG inline button callbacks (rag:action:... or rag:protocol:...)."""
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split(":")
    if len(parts) < 5:
        await query.message.reply_text("❌ Некорректные данные кнопки.")
        return

    _, rag_type, audio_class, lat_str, lon_str = parts[:5]
    lat = float(lat_str)
    lon = float(lon_str)
    class_ru = CLASS_NAME_RU.get(audio_class, audio_class)

    try:
        from cloud.agent.rag_agent import query_action, query_protocol

        await query.message.reply_text("⏳ Запрашиваю рекомендации...")

        if rag_type == "action":
            result = await query_action(audio_class, lat, lon)
        elif rag_type == "protocol":
            result = await query_protocol(audio_class, lat, lon)
        else:
            await query.message.reply_text("❌ Неизвестный тип запроса.")
            return

        await query.message.reply_text(
            f"📋 *{class_ru}* — {rag_type}\n\n{result}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception("RAG query failed")
        await query.message.reply_text(f"❌ Ошибка RAG: {e}")


def get_handlers() -> list:
    """Return all handlers to register on the Application."""
    return [
        CommandHandler("start", start),
        CommandHandler("status", status),
        CommandHandler("stop", stop),
        CommandHandler("test", test_alert),
        CallbackQueryHandler(district_chosen, pattern=r"^district:"),
        CallbackQueryHandler(rag_callback, pattern=r"^rag:"),
        MessageHandler(filters.PHOTO, handle_inspector_photo),
    ]
