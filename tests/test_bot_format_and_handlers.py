"""P2 tests: alert format, escape_markdown, handler registration.

8 tests covering formatting correctness and handler wiring.
"""

import os
import re
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

if "RANGERS_DB_PATH" not in os.environ:
    _tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    _tmp.close()
    os.environ.setdefault("RANGERS_DB_PATH", _tmp.name)

from cloud.db.incidents import clear_all_incidents
from cloud.notify.telegram import send_pending_to_chat, _last_sent
from cloud.notify.drone_bot_handlers import escape_markdown
from cloud.notify.bot_handlers import get_handlers


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    clear_all_incidents()
    _last_sent.clear()
    monkeypatch.setattr("cloud.notify.telegram.BOT_TOKEN", "test-token")
    yield
    clear_all_incidents()
    _last_sent.clear()


# ===================================================================
# TestAlertFormat (3 tests)
# ===================================================================


class TestAlertFormat:
    @pytest.fixture()
    def mock_bot(self, monkeypatch):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        monkeypatch.setattr("cloud.notify.telegram.Bot", lambda token: bot)
        return bot

    @pytest.mark.asyncio
    async def test_alert_has_msk_timestamp(self, mock_bot):
        await send_pending_to_chat(
            chat_id=6001,
            lat=57.3,
            lon=44.8,
            audio_class="chainsaw",
            reason="test",
            confidence=0.85,
            gating_level="alert",
        )

        text = mock_bot.send_message.call_args.kwargs["text"]
        assert re.search(r"\d{2}:\d{2}:\d{2} МСК", text)

    @pytest.mark.asyncio
    async def test_confidence_zero_shows_dashes(self, mock_bot):
        await send_pending_to_chat(
            chat_id=6002,
            lat=57.3,
            lon=44.8,
            audio_class="chainsaw",
            reason="test",
            confidence=0.0,
            gating_level="alert",
        )

        text = mock_bot.send_message.call_args.kwargs["text"]
        assert "---" in text

    @pytest.mark.asyncio
    async def test_keyboard_has_snooze_button(self, mock_bot):
        await send_pending_to_chat(
            chat_id=6003,
            lat=57.3,
            lon=44.8,
            audio_class="chainsaw",
            reason="test",
            confidence=0.85,
            gating_level="alert",
        )

        reply_markup = mock_bot.send_message.call_args.kwargs["reply_markup"]
        all_buttons = [btn for row in reply_markup.inline_keyboard for btn in row]
        snooze_buttons = [
            b for b in all_buttons if b.callback_data and "snooze:" in b.callback_data
        ]
        assert len(snooze_buttons) == 1


# ===================================================================
# TestEscapeMarkdown (3 tests)
# ===================================================================


class TestEscapeMarkdown:
    def test_escape_underscores_stars(self):
        assert escape_markdown("_text_") == r"\_text\_"
        assert escape_markdown("*bold*") == r"\*bold\*"

    def test_escape_backtick(self):
        assert escape_markdown("`code`") == r"\`code\`"

    def test_escape_empty_string(self):
        assert escape_markdown("") == ""


# ===================================================================
# TestHandlerRegistration (2 tests)
# ===================================================================


class TestHandlerRegistration:
    def test_get_handlers_count_18(self):
        handlers = get_handlers()
        assert len(handlers) == 18

    def test_text_handler_is_last(self):
        from telegram.ext import MessageHandler

        handlers = get_handlers()
        last = handlers[-1]
        assert isinstance(last, MessageHandler)
