"""Tests for the Drone Bot photo → vision → alert pipeline.

All external services (Vision, YandexGPT, Telegram) are mocked.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# cloud.interface.main requires FastAPI (not in test env) — register mock module
# This must happen BEFORE importing drone_bot_handlers so that its module-level
# `from cloud.interface.main import broadcast` import succeeds.
import cloud.interface

_mock_main_module = MagicMock()
_mock_main_module.broadcast = AsyncMock()
sys.modules["cloud.interface.main"] = _mock_main_module
cloud.interface.main = _mock_main_module

from cloud.notify.drone_bot_handlers import drone_start, drone_photo_handler


def _make_photo_update(chat_id: int):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    mock_photo = MagicMock()
    mock_photo.get_file = AsyncMock()
    mock_photo.get_file.return_value.download_as_bytearray = AsyncMock(
        return_value=bytearray(b"fake-photo")
    )
    update.message.photo = [mock_photo]
    return update


class TestDroneBot:
    @pytest.mark.asyncio
    async def test_start_command(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()

        await drone_start(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "drone bot" in text.lower() or "фото" in text.lower()

    @pytest.mark.asyncio
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_photo_no_threat(self, mock_classify, mock_broadcast):
        result = MagicMock()
        result.description = "Лес и тропинка"
        result.has_felling = False
        result.has_human = False
        result.has_fire = False
        mock_classify.return_value = result

        update = _make_photo_update(100)
        await drone_photo_handler(update, MagicMock())

        mock_classify.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "нарушений не обнаружено" in text.lower()
        # broadcast: vision_classified, drone_photo, pipeline_end
        assert mock_broadcast.call_count == 3

    @pytest.mark.asyncio
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch("cloud.notify.telegram.send_confirmed", new_callable=AsyncMock)
    @patch(
        "cloud.agent.decision.compose_alert",
        new_callable=AsyncMock,
        return_value="Alert text",
    )
    @patch("cloud.notify.telegram.send_pending", new_callable=AsyncMock)
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_photo_with_threat_creates_incident(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_send_confirmed,
        mock_broadcast,
    ):
        result = MagicMock()
        result.description = "Видна рубка деревьев"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        mock_classify.return_value = result
        mock_send_pending.return_value = MagicMock()

        update = _make_photo_update(200)
        await drone_photo_handler(update, MagicMock())

        mock_classify.assert_called_once()
        mock_send_pending.assert_called_once()
        assert mock_send_pending.call_args.kwargs.get("is_demo") is True
        mock_compose_alert.assert_called_once()
        mock_send_confirmed.assert_called_once()
        assert mock_send_confirmed.call_args[0][1] == b"fake-photo"
        # broadcast: vision_classified, drone_photo, alert_sent, pipeline_end
        assert mock_broadcast.call_count == 4
        text = update.message.reply_text.call_args[0][0]
        assert "инцидент создан" in text.lower()

    @pytest.mark.asyncio
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch(
        "cloud.agent.decision.compose_alert",
        new_callable=AsyncMock,
        side_effect=Exception("YandexGPT timeout"),
    )
    @patch("cloud.notify.telegram.send_pending", new_callable=AsyncMock)
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_photo_with_threat_compose_alert_fails(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_broadcast,
    ):
        """compose_alert fails but user still gets classification result."""
        result = MagicMock()
        result.description = "Видна рубка деревьев"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        mock_classify.return_value = result
        mock_send_pending.return_value = MagicMock()

        update = _make_photo_update(400)
        await drone_photo_handler(update, MagicMock())

        mock_classify.assert_called_once()
        mock_send_pending.assert_called_once()
        mock_compose_alert.assert_called_once()
        # User gets a response with threat info, NOT an error message
        text = update.message.reply_text.call_args[0][0]
        assert "угроза" in text.lower() or "chainsaw" in text.lower()
        assert "не удалось проанализировать" not in text.lower()

    @pytest.mark.asyncio
    @patch(
        "cloud.vision.classifier.classify_photo",
        new_callable=AsyncMock,
        side_effect=Exception("API error"),
    )
    async def test_photo_vision_error(self, mock_classify):
        update = _make_photo_update(300)
        await drone_photo_handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "не удалось" in text.lower()
