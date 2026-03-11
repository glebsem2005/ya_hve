"""Telegram bot Application lifecycle management.

Provides start/stop functions for use from FastAPI lifespan.
Uses the lower-level start()/updater.start_polling()/stop() sequence
instead of run_polling(), so the bot coexists with uvicorn's event loop.
"""

import os
import logging

from telegram.ext import Application

from cloud.notify.bot_handlers import get_handlers

logger = logging.getLogger(__name__)

_application: Application | None = None


def build_application() -> Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    for handler in get_handlers():
        app.add_handler(handler)
    return app


async def start_bot() -> None:
    global _application
    _application = build_application()
    await _application.initialize()
    await _application.start()
    await _application.updater.start_polling(drop_pending_updates=True)
    logger.warning("Telegram bot polling started")


async def stop_bot() -> None:
    global _application
    if _application:
        await _application.updater.stop()
        await _application.stop()
        await _application.shutdown()
        logger.warning("Telegram bot polling stopped")
        _application = None
