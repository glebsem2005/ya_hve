"""Telegram bot handlers for ranger self-registration and alerts.

Commands:
  /start  - Begin registration or show welcome if already registered
  /status - Show current registration details
  /stop   - Deactivate alerts (ranger remains in DB but active=False)
  /test   - Send a test alert with inline buttons for demo

Workflow callbacks:
  accept:<incident_id>              - Ranger accepts a call
  verdict:confirmed:<incident_id>   - Violation confirmed on site
  verdict:false:<incident_id>       - False alarm

Message handlers:
  PHOTO    - Ranger sends evidence photo (or standalone photo for Vision)
  VOICE    - Ranger sends voice description (STT -> text)
  LOCATION - Ranger shares location (proximity check)
"""

import base64
import logging
import math
import random
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from cloud.db.rangers import (
    add_ranger,
    get_ranger_by_chat_id,
    set_active,
    update_position,
)
from cloud.db.incidents import (
    get_incident,
    get_active_incident_for_chat,
    assign_chat_to_incident,
    clear_chat_incident,
    update_status,
    update_incident,
)
from cloud.notify.districts import DISTRICTS
from cloud.notify.telegram import (
    send_pending_to_chat,
    send_drone_photo,
    send_arrival_question,
    send_evidence_request,
    send_protocol_pdf,
    CLASS_NAME_RU,
    BOT_TOKEN,
)

logger = logging.getLogger(__name__)

# ---------- Registration state (manual, no ConversationHandler) ----------

_registration_state: dict[int, dict] = {}
_REG_STEP_NAME = "awaiting_name"
_REG_STEP_BADGE = "awaiting_badge"
_REG_TTL = 1800  # 30 minutes


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------- /start, /status, /stop ----------


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
        await query.edit_message_text("Неизвестное лесничество.")
        return

    chat_id = query.message.chat_id

    if get_ranger_by_chat_id(chat_id):
        await query.edit_message_text("Вы уже зарегистрированы! /status")
        return

    _registration_state[chat_id] = {
        "step": _REG_STEP_NAME,
        "district_slug": slug,
        "started_at": time.time(),
    }
    await query.edit_message_text(
        f"Лесничество: {district.name_ru}\n\nВведите ваше ФИО (фамилия, имя, отчество):"
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
    badge_line = (
        f"Табельный номер: {ranger.badge_number}\n" if ranger.badge_number else ""
    )
    await update.message.reply_text(
        f"ФИО: {ranger.name}\n"
        f"{badge_line}"
        f"Зона: {ranger.zone_lat_min:.2f}--{ranger.zone_lat_max:.2f} N, "
        f"{ranger.zone_lon_min:.2f}--{ranger.zone_lon_max:.2f} E\n"
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


# ---------- /test ----------


async def test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /test — send a demo alert with random Varnavino coordinates."""
    import random

    chat_id = update.effective_chat.id
    # Random coordinates within Varnavino forestry district
    lat = round(random.uniform(57.05, 57.55), 6)
    lon = round(random.uniform(44.60, 45.40), 6)
    classes = ["chainsaw", "gunshot", "engine", "axe"]
    audio_class = random.choice(classes)
    confidence = round(random.uniform(0.65, 0.98), 2)

    await update.message.reply_text("Отправляю тестовый алерт...")
    await send_pending_to_chat(
        chat_id=chat_id,
        lat=lat,
        lon=lon,
        audio_class=audio_class,
        reason="Test alert",
        confidence=confidence,
        gating_level="alert",
        is_demo=True,
    )


# ---------- Accept callback ----------


async def accept_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle 'Принять вызов' button."""
    query = update.callback_query
    await query.answer()

    parts = query.data.split(":", 1)
    if len(parts) < 2:
        return

    incident_id = parts[1]
    incident = get_incident(incident_id)

    if not incident:
        await query.edit_message_text("Инцидент не найден.")
        return

    if incident.status != "pending":
        await query.edit_message_text(
            f"Вызов уже принят: {incident.accepted_by_name or 'другой инспектор'}."
        )
        return

    chat_id = query.message.chat_id
    ranger = get_ranger_by_chat_id(chat_id)
    name = ranger.name if ranger else (query.from_user.full_name or str(chat_id))

    # Update incident — persist all fields
    now = time.time()
    update_incident(
        incident_id,
        status="accepted",
        accepted_by_chat_id=chat_id,
        accepted_by_name=name,
        accepted_at=now,
    )
    incident.status = "accepted"
    incident.accepted_by_chat_id = chat_id
    incident.accepted_by_name = name
    incident.accepted_at = now
    assign_chat_to_incident(chat_id, incident_id)

    maps_url = f"https://maps.yandex.ru/?pt={incident.lon},{incident.lat}&z=15"

    # Confirm to the accepting ranger
    await query.edit_message_text(f"Вызов принят. Выезжайте на точку:\n{maps_url}")

    # Edit alert for OTHER rangers (remove buttons, show who accepted)
    from telegram import Bot

    bot = Bot(token=BOT_TOKEN)
    for other_chat_id, msg_id in incident.alert_message_ids.items():
        if other_chat_id == chat_id:
            continue
        try:
            class_ru = CLASS_NAME_RU.get(incident.audio_class, incident.audio_class)
            await bot.edit_message_text(
                chat_id=other_chat_id,
                message_id=msg_id,
                text=(f"*АЛЕРТ: {class_ru}*\n━━━━━━━━━━━━━━━━\nВызов принял: {name}"),
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning("Failed to edit alert for chat %s: %s", other_chat_id, e)

    # Send drone photo if available
    await send_drone_photo(chat_id, incident)

    # Ask to share location
    await bot.send_message(
        chat_id=chat_id,
        text="Отправьте геолокацию, когда будете рядом с точкой.",
    )


# ---------- Location handler ----------


async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle shared location — check proximity to incident."""
    chat_id = update.effective_chat.id
    incident = get_active_incident_for_chat(chat_id)

    if not incident:
        return

    if incident.status not in ("accepted",):
        return

    loc = update.message.location
    dist = _haversine(loc.latitude, loc.longitude, incident.lat, incident.lon)

    PROXIMITY_RADIUS_M = 1000

    if incident.is_demo or dist <= PROXIMITY_RADIUS_M:
        now = time.time()
        resp_min = None
        if incident.created_at:
            resp_min = round((now - incident.created_at) / 60, 1)
        update_incident(
            incident.id,
            status="on_site",
            arrived_at=now,
            response_time_min=resp_min,
        )
        incident.status = "on_site"
        incident.arrived_at = now
        incident.response_time_min = resp_min
        await send_arrival_question(chat_id, incident)
    else:
        await update.message.reply_text(
            f"Вы в {dist:.0f} м от точки. Продолжайте движение."
        )


# ---------- Verdict callback ----------


async def verdict_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle 'Нарушение подтверждено' / 'Ложный вызов' buttons."""
    query = update.callback_query
    await query.answer()

    parts = query.data.split(":")
    if len(parts) < 3:
        return

    _, verdict_type, incident_id = parts[:3]
    incident = get_incident(incident_id)

    if not incident:
        await query.edit_message_text("Инцидент не найден.")
        return

    chat_id = query.message.chat_id

    if verdict_type == "false":
        update_incident(
            incident_id,
            status="false_alarm",
            resolution_details="Ложное срабатывание, закрыто инспектором",
        )
        if incident:
            incident.status = "false_alarm"
        clear_chat_incident(chat_id)
        await query.edit_message_text("Принято, инцидент закрыт. Спасибо за проверку.")

    elif verdict_type == "confirmed":
        # Stay on_site, wait for evidence
        await query.edit_message_text("Нарушение зафиксировано.")
        await send_evidence_request(chat_id)


# ---------- Voice handler ----------


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice message — STT and save as ranger report."""
    chat_id = update.effective_chat.id
    incident = get_active_incident_for_chat(chat_id)

    if not incident or incident.status != "on_site":
        await update.message.reply_text("Нет активного инцидента для записи.")
        return

    await update.message.reply_text("Распознаю голосовое сообщение...")

    try:
        voice_file = await update.message.voice.get_file()
        voice_bytes = await voice_file.download_as_bytearray()

        from cloud.agent.stt import recognize_voice

        text = await recognize_voice(bytes(voice_bytes))

        if not text:
            await update.message.reply_text(
                "Не удалось распознать голос. Опишите ситуацию текстом."
            )
            return

        incident.ranger_report_raw = text
        update_incident(incident.id, ranger_report_raw=text)
        await update.message.reply_text(f'Текст сохранен:\n"{text}"')

        # If photo already collected, generate protocol
        if incident.ranger_photo_b64:
            await _generate_and_send_protocol(chat_id, incident)

    except Exception as e:
        logger.exception("Voice handler failed")
        await update.message.reply_text("Ошибка обработки голосового сообщения.")


# ---------- Photo handler ----------


async def handle_inspector_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle photo from inspector.

    If there's an active on_site incident, save as evidence.
    Otherwise, classify via YandexGPT Vision (standalone mode).
    """
    chat_id = update.effective_chat.id
    incident = get_active_incident_for_chat(chat_id)

    if incident and incident.status == "on_site":
        # Evidence collection mode
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            incident.ranger_photo_b64 = base64.b64encode(photo_bytes).decode()

            # Check for caption as report text
            if update.message.caption:
                incident.ranger_report_raw = update.message.caption
                update_incident(incident.id, ranger_report_raw=update.message.caption)

            await update.message.reply_text("Фото сохранено.")

            if incident.ranger_report_raw:
                await _generate_and_send_protocol(chat_id, incident)
            else:
                await update.message.reply_text(
                    "Опишите нарушение (текстом или голосовым сообщением)."
                )
        except Exception as e:
            logger.exception("Evidence photo save failed")
            await update.message.reply_text("Ошибка сохранения фото.")
        return

    # Standalone Vision classification → full drone pipeline for demo
    await update.message.reply_text("Анализирую фото...")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = bytes(await photo_file.download_as_bytearray())
        photo_b64 = base64.b64encode(photo_bytes).decode()

        from cloud.vision.classifier import classify_photo

        result = await classify_photo(photo_b64)

        # Broadcast vision result to web dashboard
        try:
            from cloud.interface.main import broadcast
        except ImportError:

            async def broadcast(_msg):
                pass

        await broadcast(
            {
                "type": "vision_classified",
                "description": result.description,
                "has_felling": result.has_felling,
                "has_human": result.has_human,
                "has_fire": result.has_fire,
            }
        )
        await broadcast({"type": "drone_photo", "photo_b64": photo_b64})

        has_threat = result.has_felling or result.has_human or result.has_fire

        if has_threat:
            # Map vision flags to audio class for pipeline compatibility
            if result.has_felling:
                audio_class = "chainsaw"
            elif result.has_fire:
                audio_class = "fire"
            else:
                audio_class = "engine"

            # Get coordinates: ranger position or random in Varnavino
            ranger = get_ranger_by_chat_id(chat_id)
            if ranger and ranger.current_lat and ranger.current_lon:
                lat, lon = ranger.current_lat, ranger.current_lon
            else:
                lat = round(random.uniform(57.05, 57.55), 6)
                lon = round(random.uniform(44.60, 45.40), 6)

            confidence = 0.85

            # 1. Create incident + alert rangers
            from cloud.notify.telegram import send_pending

            incident = await send_pending(
                lat=lat,
                lon=lon,
                audio_class=audio_class,
                reason=result.description,
                confidence=confidence,
                is_demo=True,
            )

            # 2. Compose alert text via YandexGPT
            from cloud.agent.decision import compose_alert

            alert = await compose_alert(
                audio_class=audio_class,
                visual_description=result.description,
                lat=lat,
                lon=lon,
                confidence=confidence,
            )

            # 3. Store drone photo in incident (sent after accept)
            from cloud.notify.telegram import send_confirmed

            await send_confirmed(alert, photo_bytes, incident)

            await broadcast(
                {
                    "type": "alert_sent",
                    "audio_class": audio_class,
                    "lat": lat,
                    "lon": lon,
                }
            )
            await broadcast({"type": "pipeline_end", "result": "incident_created"})

            await update.message.reply_text(
                f"*Инцидент создан*\n\n"
                f"Класс: {audio_class}\n"
                f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n"
                f"Алерт отправлен инспекторам.",
                parse_mode="Markdown",
            )
        else:
            await broadcast({"type": "pipeline_end", "result": "no_threat"})

            reply = f"*Анализ фото:*\n{result.description}\n\nНарушений не обнаружено."
            await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as e:
        logger.exception("Photo classification failed")
        await update.message.reply_text(
            "Не удалось проанализировать фото. Попробуйте ещё раз или опишите ситуацию текстом."
        )


# ---------- Text handler (for on_site report without voice) ----------


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text message — registration flow or ranger report if on_site."""
    chat_id = update.effective_chat.id
    text = update.message.text
    if not text or text.startswith("/"):
        return

    # --- Registration flow ---
    reg = _registration_state.get(chat_id)
    if reg:
        elapsed = time.time() - reg["started_at"]
        if elapsed > _REG_TTL:
            _registration_state.pop(chat_id, None)
            await update.message.reply_text(
                "Регистрация просрочена. Отправьте /start чтобы начать заново."
            )
            return

        if reg["step"] == _REG_STEP_NAME:
            name = text.strip()
            if len(name.split()) < 2:
                await update.message.reply_text(
                    "Введите полное ФИО (минимум фамилия и имя)."
                )
                return
            reg["name"] = name
            reg["step"] = _REG_STEP_BADGE
            await update.message.reply_text("Введите ваш табельный номер:")
            return

        if reg["step"] == _REG_STEP_BADGE:
            badge = text.strip()
            if not badge:
                await update.message.reply_text("Табельный номер не может быть пустым.")
                return

            slug = reg["district_slug"]
            district = DISTRICTS.get(slug)
            if not district:
                _registration_state.pop(chat_id, None)
                await update.message.reply_text(
                    "Ошибка регистрации. Попробуйте /start."
                )
                return

            try:
                add_ranger(
                    name=reg["name"],
                    chat_id=chat_id,
                    badge_number=badge,
                    zone_lat_min=district.lat_min,
                    zone_lat_max=district.lat_max,
                    zone_lon_min=district.lon_min,
                    zone_lon_max=district.lon_max,
                )
            except Exception:
                logger.exception("Failed to register ranger chat_id=%s", chat_id)
                _registration_state.pop(chat_id, None)
                await update.message.reply_text("Ошибка регистрации. Попробуйте позже.")
                return

            # Assign random position within the district
            rand_lat = round(random.uniform(district.lat_min, district.lat_max), 6)
            rand_lon = round(random.uniform(district.lon_min, district.lon_max), 6)
            update_position(chat_id, rand_lat, rand_lon)

            _registration_state.pop(chat_id, None)
            await update.message.reply_text(
                f"Вы зарегистрированы!\n\n"
                f"ФИО: {reg['name']}\n"
                f"Табельный номер: {badge}\n"
                f"Лесничество: {district.name_ru}\n"
                f"Ваша позиция: {rand_lat:.4f} N, {rand_lon:.4f} E\n\n"
                "Вы будете получать оповещения о подозрительной активности "
                "в вашей зоне. Используйте /stop для отключения."
            )
            return

    # --- On-site report ---
    incident = get_active_incident_for_chat(chat_id)

    if not incident or incident.status != "on_site":
        return  # Ignore non-contextual text

    incident.ranger_report_raw = text
    update_incident(incident.id, ranger_report_raw=text)
    await update.message.reply_text("Описание сохранено.")

    if incident.ranger_photo_b64:
        await _generate_and_send_protocol(chat_id, incident)
    else:
        await update.message.reply_text("Отправьте фото нарушения.")


# ---------- RAG callback ----------


async def rag_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle RAG inline button callbacks (rag:action:... or rag:protocol:...)."""
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split(":")
    if len(parts) < 5:
        await query.message.reply_text("Некорректные данные кнопки.")
        return

    _, rag_type, audio_class, lat_str, lon_str = parts[:5]
    lat = float(lat_str)
    lon = float(lon_str)
    class_ru = CLASS_NAME_RU.get(audio_class, audio_class)

    try:
        from cloud.agent.rag_agent import query_action, query_protocol

        await query.message.reply_text("Запрашиваю рекомендации...")

        if rag_type == "action":
            result = await query_action(audio_class, lat, lon)
        elif rag_type == "protocol":
            result = await query_protocol(audio_class, lat, lon)
        else:
            await query.message.reply_text("Неизвестный тип запроса.")
            return

        await query.message.reply_text(
            f"*{class_ru}* -- {rag_type}\n\n{result}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception("RAG query failed")
        await query.message.reply_text(f"Ошибка RAG: {e}")


# ---------- Protocol generation ----------


async def _generate_and_send_protocol(chat_id: int, incident) -> None:
    """Generate legal text via YandexGPT, get legal articles via RAG, build PDF."""
    from telegram import Bot

    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=chat_id, text="Формирую протокол...")

    # 1. YandexGPT: raw report -> legal language
    try:
        from cloud.agent.rag_agent import legalize_report

        legal_text = await legalize_report(
            incident.audio_class, incident.ranger_report_raw
        )
        incident.ranger_report_legal = legal_text
        update_incident(incident.id, ranger_report_legal=legal_text)
    except Exception as e:
        logger.warning("Failed to legalize report via YandexGPT: %s", e)
        incident.ranger_report_legal = incident.ranger_report_raw
        update_incident(incident.id, ranger_report_legal=incident.ranger_report_raw)
        await bot.send_message(
            chat_id=chat_id,
            text="Не удалось обработать описание через YandexGPT, "
            "используется исходный текст.",
        )

    # 2. RAG: get applicable legal articles only
    legal_articles = ""
    try:
        from cloud.agent.rag_agent import query_legal_articles

        legal_articles = await query_legal_articles(
            incident.audio_class, incident.lat, incident.lon
        )
    except Exception as e:
        logger.warning("RAG query for legal articles failed: %s", e)

    # 3. Generate PDF
    try:
        from cloud.agent.protocol_pdf import generate_protocol

        pdf_bytes = generate_protocol(incident, legal_articles)
        incident.protocol_pdf = pdf_bytes
    except Exception as e:
        logger.exception("PDF generation failed")
        await bot.send_message(
            chat_id=chat_id,
            text="Ошибка генерации PDF-протокола.",
        )
        return

    # 4. Send PDF and resolve
    await send_protocol_pdf(chat_id, pdf_bytes)
    update_incident(
        incident.id,
        status="resolved",
        resolution_details="Протокол составлен, материалы переданы",
    )
    incident.status = "resolved"
    clear_chat_incident(chat_id)


# ---------- Handler registration ----------


def get_handlers() -> list:
    """Return all handlers to register on the Application."""
    return [
        CommandHandler("start", start),
        CommandHandler("status", status),
        CommandHandler("stop", stop),
        CommandHandler("test", test_alert),
        CallbackQueryHandler(district_chosen, pattern=r"^district:"),
        CallbackQueryHandler(accept_callback, pattern=r"^accept:"),
        CallbackQueryHandler(verdict_callback, pattern=r"^verdict:"),
        CallbackQueryHandler(rag_callback, pattern=r"^rag:"),
        MessageHandler(filters.VOICE, voice_handler),
        MessageHandler(filters.LOCATION, location_handler),
        MessageHandler(filters.PHOTO, handle_inspector_photo),
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler),
    ]
