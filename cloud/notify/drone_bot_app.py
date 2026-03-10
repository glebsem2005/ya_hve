"""Drone Bot Application lifecycle management.

Separate Telegram bot for receiving drone photos.
Uses the same polling approach as the Ranger bot (bot_app.py).
"""

import os
import logging

from telegram.ext import Application

from cloud.notify.drone_bot_handlers import get_drone_handlers

logger = logging.getLogger(__name__)

_application: Application | None = None


def build_drone_application() -> Application:
    token = os.getenv("TELEGRAM_DRONE_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_DRONE_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    for handler in get_drone_handlers():
        app.add_handler(handler)
    return app


async def start_drone_bot() -> None:
    global _application
    _application = build_drone_application()
    await _application.initialize()
    await _application.start()
    await _application.updater.start_polling(drop_pending_updates=True)
    logger.info("Drone bot polling started")


async def stop_drone_bot() -> None:
    global _application
    if _application:
        await _application.updater.stop()
        await _application.stop()
        await _application.shutdown()
        logger.info("Drone bot polling stopped")
        _application = None
