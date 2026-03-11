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

from cloud.notify.drone_bot_handlers import (
    drone_start,
    drone_photo_handler,
    drone_text_handler,
)


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
        result.is_threat = False
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
        result.is_threat = True
        mock_classify.return_value = result
        mock_send_pending.return_value = MagicMock()

        update = _make_photo_update(200)
        await drone_photo_handler(update, MagicMock())

        mock_classify.assert_called_once()
        mock_send_pending.assert_called_once()
        assert mock_send_pending.call_args.kwargs.get("is_demo") is True
        assert mock_send_pending.call_args.kwargs.get("broadcast") is True
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
        result.is_threat = True
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
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_photo_description_with_markdown_chars(
        self, mock_classify, mock_broadcast
    ):
        """No-threat path: description with Markdown special chars must not crash."""
        result = MagicMock()
        result.description = "Лес с _подлеском_ и *кустами* у [реки]"
        result.has_felling = False
        result.has_human = False
        result.has_fire = False
        result.is_threat = False
        mock_classify.return_value = result

        update = _make_photo_update(500)
        await drone_photo_handler(update, MagicMock())

        # Must NOT fall into the outer except → "Не удалось проанализировать"
        final_text = update.message.reply_text.call_args[0][0]
        assert "не удалось проанализировать" not in final_text.lower()
        assert "нарушений не обнаружено" in final_text.lower()

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
    async def test_photo_threat_description_with_markdown_chars(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_send_confirmed,
        mock_broadcast,
    ):
        """Threat path: description with [brackets] must not crash reply_text."""
        result = MagicMock()
        result.description = "Обнаружена [незаконная_рубка] деревьев *хвойных*"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        result.is_threat = True
        mock_classify.return_value = result
        mock_send_pending.return_value = MagicMock()

        update = _make_photo_update(600)
        await drone_photo_handler(update, MagicMock())

        final_text = update.message.reply_text.call_args[0][0]
        assert "не удалось проанализировать" not in final_text.lower()
        assert "угроза" in final_text.lower() or "chainsaw" in final_text.lower()

    @pytest.mark.asyncio
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_photo_reply_markdown_fallback(self, mock_classify, mock_broadcast):
        """If Markdown reply_text raises, fallback to plain text."""
        from telegram.error import BadRequest

        result = MagicMock()
        result.description = "Лес и тропинка"
        result.has_felling = False
        result.has_human = False
        result.has_fire = False
        result.is_threat = False
        mock_classify.return_value = result

        update = _make_photo_update(700)
        # First call is "Анализирую фото...", second is the Markdown reply
        # Make Markdown reply raise BadRequest, then plain text should succeed
        call_count = 0
        original_reply = update.message.reply_text

        async def reply_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("parse_mode") == "Markdown":
                raise BadRequest("Can't parse entities")
            return None

        update.message.reply_text = AsyncMock(side_effect=reply_side_effect)
        await drone_photo_handler(update, MagicMock())

        # Should have retried without parse_mode after BadRequest
        calls = update.message.reply_text.call_args_list
        # Last call must be plain text (no parse_mode)
        last_call = calls[-1]
        assert last_call.kwargs.get("parse_mode") is None
        assert "не удалось проанализировать" not in last_call[0][0].lower()

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
    async def test_threat_machinery_maps_to_engine(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_send_confirmed,
        mock_broadcast,
    ):
        """Heavy machinery + felling → audio_class='engine', not 'chainsaw'."""
        result = MagicMock()
        result.description = "Тяжёлая техника на вырубке леса"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        result.has_machinery = True
        result.is_threat = True
        mock_classify.return_value = result
        mock_send_pending.return_value = MagicMock()

        update = _make_photo_update(800)
        await drone_photo_handler(update, MagicMock())

        # send_pending should receive audio_class='engine'
        assert mock_send_pending.call_args.kwargs.get("audio_class") == "engine"
        text = update.message.reply_text.call_args[0][0]
        assert "engine" in text.lower()

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
    async def test_threat_felling_no_machinery_maps_to_chainsaw(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_send_confirmed,
        mock_broadcast,
    ):
        """Felling without machinery → audio_class='chainsaw' (existing behavior)."""
        result = MagicMock()
        result.description = "Рубка деревьев бензопилой"
        result.has_felling = True
        result.has_human = True
        result.has_fire = False
        result.has_machinery = False
        result.is_threat = True
        mock_classify.return_value = result
        mock_send_pending.return_value = MagicMock()

        update = _make_photo_update(900)
        await drone_photo_handler(update, MagicMock())

        assert mock_send_pending.call_args.kwargs.get("audio_class") == "chainsaw"

    @pytest.mark.asyncio
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_broadcast_includes_has_machinery(
        self, mock_classify, mock_broadcast
    ):
        """vision_classified broadcast must include has_machinery field."""
        result = MagicMock()
        result.description = "Харвестер на вырубке"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        result.has_machinery = True
        result.is_threat = False
        mock_classify.return_value = result

        update = _make_photo_update(1000)
        await drone_photo_handler(update, MagicMock())

        # Find the vision_classified broadcast call
        vision_call = None
        for call in mock_broadcast.call_args_list:
            msg = call[0][0]
            if isinstance(msg, dict) and msg.get("event") == "vision_classified":
                vision_call = msg
                break
        assert vision_call is not None, "vision_classified broadcast not found"
        assert "has_machinery" in vision_call
        assert vision_call["has_machinery"] is True

    @pytest.mark.asyncio
    @patch("cloud.notify.drone_bot_handlers.broadcast", new_callable=AsyncMock)
    @patch(
        "cloud.agent.decision.compose_alert",
        new_callable=AsyncMock,
        return_value="Alert text",
    )
    @patch(
        "cloud.notify.telegram.send_pending",
        new_callable=AsyncMock,
        side_effect=Exception("Telegram down"),
    )
    @patch("cloud.vision.classifier.classify_photo", new_callable=AsyncMock)
    async def test_threat_skips_compose_when_pending_fails(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_broadcast,
    ):
        """If send_pending fails, compose_alert should NOT be called."""
        result = MagicMock()
        result.description = "Рубка деревьев"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        result.has_machinery = False
        result.is_threat = True
        mock_classify.return_value = result

        update = _make_photo_update(1100)
        await drone_photo_handler(update, MagicMock())

        mock_send_pending.assert_called_once()
        mock_compose_alert.assert_not_called()

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
    async def test_drone_bot_broadcast_includes_incident_id(
        self,
        mock_classify,
        mock_send_pending,
        mock_compose_alert,
        mock_send_confirmed,
        mock_broadcast,
    ):
        """alert_sent broadcast must include incident_id."""
        result = MagicMock()
        result.description = "Рубка деревьев"
        result.has_felling = True
        result.has_human = False
        result.has_fire = False
        result.has_machinery = False
        result.is_threat = True
        mock_classify.return_value = result
        incident_mock = MagicMock()
        incident_mock.id = "inc-42"
        mock_send_pending.return_value = incident_mock

        update = _make_photo_update(1200)
        await drone_photo_handler(update, MagicMock())

        # Find the alert_sent broadcast
        alert_call = None
        for call in mock_broadcast.call_args_list:
            msg = call[0][0]
            if isinstance(msg, dict) and msg.get("event") == "alert_sent":
                alert_call = msg
                break
        assert alert_call is not None, "alert_sent broadcast not found"
        assert "incident_id" in alert_call
        assert alert_call["incident_id"] == "inc-42"

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

    @pytest.mark.asyncio
    async def test_text_message_replies_with_photo_hint(self):
        """Non-photo text message must get a reply hinting to send a photo."""
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.text = "привет"

        await drone_text_handler(update, MagicMock())

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "фото" in text.lower()
