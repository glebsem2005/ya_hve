"""Drone Bot Application lifecycle management.

Separate Telegram bot for receiving drone photos.
Uses the same polling approach as the Ranger bot (bot_app.py).
"""

import os
import logging
import traceback

from telegram.ext import Application, ContextTypes

from cloud.notify.drone_bot_handlers import get_drone_handlers

logger = logging.getLogger(__name__)

_application: Application | None = None


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — log traceback, notify user."""
    logger.error(
        "Drone bot exception:\n%s",
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


def build_drone_application() -> Application:
    token = os.getenv("TELEGRAM_DRONE_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_DRONE_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    for handler in get_drone_handlers():
        app.add_handler(handler)
    app.add_error_handler(_error_handler)
    return app


async def start_drone_bot() -> None:
    global _application
    _application = build_drone_application()
    await _application.initialize()
    await _application.start()
    await _application.updater.start_polling(drop_pending_updates=True)
    logger.warning("Drone bot polling started")


async def stop_drone_bot() -> None:
    global _application
    if _application:
        await _application.updater.stop()
        await _application.stop()
        await _application.shutdown()
        logger.warning("Drone bot polling stopped")
        _application = None
