import os
from telegram import Bot
from telegram.constants import ParseMode
from cloud.agent.decision import Alert

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

async def send_pending(lat: float, lon: float, audio_class: str, reason: str) -> None:
    bot = Bot(token=BOT_TOKEN)
    maps_link = f"https://maps.yandex.ru/?pt={lon},{lat}&z=15"

    text = (
        f"*Обнаружена аномалия*\n\n"
        f"Звук: `{audio_class}`\n"
        f"[{lat:.4f}°N, {lon:.4f}°E]({maps_link})\n\n"
        f"Дрон вылетел для подтверждения..."
    )

    await bot.send_message(
        chat_id=CHAT_ID,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )

async def send_confirmed(alert: Alert, photo_bytes: bytes) -> None:
    bot = Bot(token=BOT_TOKEN)
    maps_link = f"https://maps.yandex.ru/?pt={alert.lon},{alert.lat}&z=15"

    caption = (
        f"{alert.priority}\n\n"
        f"{alert.text}\n\n"
        f"[{alert.lat:.4f}°N, {alert.lon:.4f}°E]({maps_link})"
    )

    await bot.send_photo(
        chat_id=CHAT_ID,
        photo=photo_bytes,
        caption=caption,
        parse_mode=ParseMode.MARKDOWN,
    )
