"""Telegram bot Application lifecycle management.

Provides start/stop functions for use from FastAPI lifespan.
Uses the lower-level start()/updater.start_polling()/stop() sequence
instead of run_polling(), so the bot coexists with uvicorn's event loop.
"""

import os
import logging
import traceback

from telegram import BotCommand, MenuButtonWebApp, WebAppInfo
from telegram.ext import Application, ContextTypes

from cloud.notify.bot_handlers import get_handlers
from cloud.db.incidents import get_stale_incidents, update_incident, clear_chat_incident

logger = logging.getLogger(__name__)

_application: Application | None = None


async def _post_init(application: Application) -> None:
    """Set bot commands and Mini App menu button after init."""
    bot = application.bot
    try:
        await bot.set_my_commands(
            [
                BotCommand("start", "Регистрация / активация"),
                BotCommand("status", "Статус регистрации"),
                BotCommand("stop", "Отключить оповещения"),
                BotCommand("test", "Тестовый алерт"),
                BotCommand("help", "Справка по командам"),
                BotCommand("cancel", "Отменить регистрацию"),
                BotCommand("rangers", "Список инспекторов (админ)"),
            ]
        )
        logger.warning("Bot commands set OK")
    except Exception as e:
        logger.warning("Failed to set bot commands: %s", e)

    try:
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="Карта",
                web_app=WebAppInfo(url="https://faun-forrest.duckdns.org/"),
            )
        )
        logger.warning("Menu button set OK")
    except Exception as e:
        logger.warning("Failed to set menu button: %s", e)


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — log traceback, notify user."""
    logger.error(
        "Bot exception:\n%s",
        "".join(traceback.format_exception(context.error)),
    )
    if update and hasattr(update, "effective_chat") and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла ошибка. Попробуйте позже.",
            )
        except Exception:
            pass


async def _cleanup_stale_incidents(context: ContextTypes.DEFAULT_TYPE) -> None:
    """JobQueue callback: close stale pending (>30m) and accepted (>60m) incidents."""
    stale = get_stale_incidents(pending_max_age=1800, accepted_max_age=3600)
    for incident in stale:
        old_status = incident.status
        update_incident(
            incident.id,
            status="false_alarm",
            resolution_details=f"Автозакрытие: {old_status} просрочен",
        )
        if incident.accepted_by_chat_id:
            clear_chat_incident(incident.accepted_by_chat_id)
        logger.info("Auto-closed stale incident %s (was %s)", incident.id, old_status)

    if stale:
        logger.info("Cleaned up %d stale incidents", len(stale))


def build_application() -> Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    for handler in get_handlers():
        app.add_handler(handler)
    app.add_error_handler(_error_handler)
    return app


async def start_bot() -> None:
    global _application
    _application = build_application()
    await _application.initialize()
    await _post_init(_application)
    await _application.start()
    await _application.updater.start_polling(drop_pending_updates=True)

    # Schedule stale incident cleanup every 5 minutes
    if _application.job_queue:
        _application.job_queue.run_repeating(
            _cleanup_stale_incidents, interval=300, first=60
        )
    else:
        logger.warning(
            "JobQueue not available — stale incident cleanup disabled. "
            "Install python-telegram-bot[job-queue] to enable."
        )

    logger.warning("Telegram bot polling started")


async def stop_bot() -> None:
    global _application
    if _application:
        await _application.updater.stop()
        await _application.stop()
        await _application.shutdown()
        logger.warning("Telegram bot polling stopped")
        _application = None
