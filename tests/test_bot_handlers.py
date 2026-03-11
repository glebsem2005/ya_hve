"""Tests for Telegram bot registration handlers.

Uses mock Update/Message objects to test handler logic without
connecting to Telegram API.
"""

import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
_tmp.close()
os.environ["RANGERS_DB_PATH"] = _tmp.name

from cloud.db.rangers import (
    init_db,
    _migrate_db,
    get_ranger_by_chat_id,
    add_ranger,
    set_active,
)
from cloud.notify.bot_handlers import (
    start,
    district_chosen,
    status,
    stop,
    text_handler,
    help_cmd,
    cancel_cmd,
    rangers_cmd,
    confirm_reg_callback,
    snooze_callback,
    _registration_state,
    _REG_STEP_NAME,
    _REG_STEP_BADGE,
    _REG_STEP_CONFIRM,
    _REG_TTL,
    ADMIN_CHAT_IDS,
)
from cloud.notify.districts import DISTRICTS


@pytest.fixture(autouse=True)
def _clean_db():
    import sqlite3

    conn = sqlite3.connect(os.environ["RANGERS_DB_PATH"])
    conn.execute("DROP TABLE IF EXISTS rangers")
    conn.commit()
    conn.close()
    init_db()
    _registration_state.clear()
    yield


def _make_update(chat_id: int = 111, full_name: str = "Тест Тестов", text: str = ""):
    """Create a mock Update with message."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    update.message.text = text or None
    # For callback queries
    update.callback_query = None
    # User info
    user = MagicMock()
    user.full_name = full_name
    user.username = "test_user"
    update.message.from_user = user
    return update


def _make_callback_update(
    chat_id: int = 111, data: str = "district:varnavino", full_name: str = "Тест Тестов"
):
    """Create a mock Update with callback query."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = data
    update.callback_query.message.chat_id = chat_id
    update.callback_query.from_user.full_name = full_name
    update.callback_query.from_user.username = "test_user"
    return update


class TestStartHandler:
    @pytest.mark.asyncio
    async def test_new_user_gets_district_keyboard(self):
        update = _make_update(chat_id=100)
        await start(update, MagicMock())

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        assert "Выберите ваше лесничество" in call_args[0][0]
        assert call_args[1]["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_existing_active_user_gets_greeting(self):
        add_ranger(
            "Иван",
            chat_id=200,
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        update = _make_update(chat_id=200)
        await start(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "уже зарегистрированы" in text

    @pytest.mark.asyncio
    async def test_inactive_user_reactivated(self):
        add_ranger(
            "Пётр",
            chat_id=300,
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        set_active(300, False)

        update = _make_update(chat_id=300)
        await start(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "С возвращением" in text
        assert get_ranger_by_chat_id(300).active is True


class TestDistrictChosen:
    @pytest.mark.asyncio
    async def test_district_chosen_asks_for_name(self):
        update = _make_callback_update(chat_id=400, data="district:varnavino")
        await district_chosen(update, MagicMock())

        # Should NOT register yet — asks for name
        assert get_ranger_by_chat_id(400) is None
        assert 400 in _registration_state
        assert _registration_state[400]["step"] == _REG_STEP_NAME

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "ФИО" in text

    @pytest.mark.asyncio
    async def test_unknown_district_shows_error(self):
        update = _make_callback_update(chat_id=500, data="district:unknown")
        await district_chosen(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "неизвестное лесничество" in text.lower()
        assert get_ranger_by_chat_id(500) is None

    @pytest.mark.asyncio
    async def test_already_registered_blocked(self):
        add_ranger(
            "Уже есть",
            chat_id=600,
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        update = _make_callback_update(chat_id=600, data="district:varnavino")
        await district_chosen(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "уже зарегистрированы" in text


class TestStatusHandler:
    @pytest.mark.asyncio
    async def test_registered_user_sees_info(self):
        add_ranger(
            "Сергей Иванов",
            chat_id=700,
            badge_number="777",
            zone_lat_min=57.05,
            zone_lat_max=57.55,
            zone_lon_min=44.60,
            zone_lon_max=45.40,
        )
        update = _make_update(chat_id=700)
        await status(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "Сергей" in text
        assert "777" in text
        assert "включены" in text

    @pytest.mark.asyncio
    async def test_unregistered_user_gets_prompt(self):
        update = _make_update(chat_id=800)
        await status(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "не зарегистрированы" in text


class TestStopHandler:
    @pytest.mark.asyncio
    async def test_active_ranger_deactivated(self):
        add_ranger(
            "Алексей",
            chat_id=900,
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        update = _make_update(chat_id=900)
        await stop(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "отключены" in text
        assert get_ranger_by_chat_id(900).active is False

    @pytest.mark.asyncio
    async def test_already_inactive_ranger(self):
        add_ranger(
            "Борис",
            chat_id=1000,
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        set_active(1000, False)
        update = _make_update(chat_id=1000)
        await stop(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "уже отключены" in text

    @pytest.mark.asyncio
    async def test_unregistered_user(self):
        update = _make_update(chat_id=1100)
        await stop(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "не зарегистрированы" in text


class TestRegistrationFlow:
    @pytest.mark.asyncio
    async def test_district_chosen_asks_for_name(self):
        update = _make_callback_update(chat_id=1200, data="district:varnavino")
        await district_chosen(update, MagicMock())

        assert 1200 in _registration_state
        assert _registration_state[1200]["step"] == _REG_STEP_NAME
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "ФИО" in text

    @pytest.mark.asyncio
    async def test_name_entered_asks_for_badge(self):
        _registration_state[1300] = {
            "step": _REG_STEP_NAME,
            "district_slug": "varnavino",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=1300, text="Иванов Иван Иванович")
        await text_handler(update, MagicMock())

        assert _registration_state[1300]["step"] == _REG_STEP_BADGE
        assert _registration_state[1300]["name"] == "Иванов Иван Иванович"
        text = update.message.reply_text.call_args[0][0]
        assert "табельный" in text.lower()

    @pytest.mark.asyncio
    async def test_badge_entered_shows_confirmation(self):
        _registration_state[1400] = {
            "step": _REG_STEP_BADGE,
            "district_slug": "varnavino",
            "name": "Петров Пётр Петрович",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=1400, text="12345")
        await text_handler(update, MagicMock())

        # Not yet registered — awaiting confirmation
        assert get_ranger_by_chat_id(1400) is None
        assert _registration_state[1400]["step"] == _REG_STEP_CONFIRM
        assert _registration_state[1400]["badge"] == "12345"
        text = update.message.reply_text.call_args[0][0]
        assert "проверьте данные" in text.lower()

    @pytest.mark.asyncio
    async def test_confirm_yes_completes_registration(self):
        _registration_state[1401] = {
            "step": _REG_STEP_CONFIRM,
            "district_slug": "varnavino",
            "name": "Петров Пётр Петрович",
            "badge": "12345",
            "started_at": time.time(),
        }
        update = _make_callback_update(chat_id=1401, data="confirm_reg:yes")
        await confirm_reg_callback(update, MagicMock())

        ranger = get_ranger_by_chat_id(1401)
        assert ranger is not None
        assert ranger.name == "Петров Пётр Петрович"
        assert ranger.badge_number == "12345"
        assert 1401 not in _registration_state
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "зарегистрированы" in text.lower()

    @pytest.mark.asyncio
    async def test_confirm_no_restarts_registration(self):
        _registration_state[1402] = {
            "step": _REG_STEP_CONFIRM,
            "district_slug": "varnavino",
            "name": "Петров Пётр Петрович",
            "badge": "12345",
            "started_at": time.time(),
        }
        update = _make_callback_update(chat_id=1402, data="confirm_reg:no")
        await confirm_reg_callback(update, MagicMock())

        assert get_ranger_by_chat_id(1402) is None
        assert 1402 not in _registration_state
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "отменена" in text.lower()

    @pytest.mark.asyncio
    async def test_short_name_rejected(self):
        _registration_state[1500] = {
            "step": _REG_STEP_NAME,
            "district_slug": "varnavino",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=1500, text="Иван")
        await text_handler(update, MagicMock())

        # Still awaiting name
        assert _registration_state[1500]["step"] == _REG_STEP_NAME
        text = update.message.reply_text.call_args[0][0]
        assert "полное ФИО" in text.lower() or "минимум" in text.lower()

    @pytest.mark.asyncio
    async def test_empty_badge_rejected(self):
        _registration_state[1600] = {
            "step": _REG_STEP_BADGE,
            "district_slug": "varnavino",
            "name": "Сидоров Сидор",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=1600, text="   ")
        await text_handler(update, MagicMock())

        # Still awaiting badge — empty after strip
        assert 1600 in _registration_state
        text = update.message.reply_text.call_args[0][0]
        assert "не может быть пустым" in text.lower() or "табельный" in text.lower()

    @pytest.mark.asyncio
    async def test_registration_timeout(self):
        _registration_state[1700] = {
            "step": _REG_STEP_NAME,
            "district_slug": "varnavino",
            "started_at": time.time() - _REG_TTL - 1,
        }
        update = _make_update(chat_id=1700, text="Иванов Иван Иванович")
        await text_handler(update, MagicMock())

        assert 1700 not in _registration_state
        text = update.message.reply_text.call_args[0][0]
        assert "просрочена" in text.lower()

    @pytest.mark.asyncio
    async def test_status_shows_badge_number(self):
        add_ranger(
            "Козлов Козёл Козлович",
            chat_id=1800,
            badge_number="99887",
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        update = _make_update(chat_id=1800)
        await status(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "99887" in text
        assert "Козлов" in text


class TestHelpCommand:
    @pytest.mark.asyncio
    async def test_help_shows_commands(self):
        update = _make_update(chat_id=2000)
        await help_cmd(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "/start" in text
        assert "/help" in text
        assert "/cancel" in text
        assert "/rangers" in text


class TestCancelCommand:
    @pytest.mark.asyncio
    async def test_cancel_active_registration(self):
        _registration_state[2100] = {
            "step": _REG_STEP_NAME,
            "district_slug": "varnavino",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=2100)
        await cancel_cmd(update, MagicMock())

        assert 2100 not in _registration_state
        text = update.message.reply_text.call_args[0][0]
        assert "отменена" in text.lower()

    @pytest.mark.asyncio
    async def test_cancel_no_registration(self):
        update = _make_update(chat_id=2200)
        await cancel_cmd(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "нет активной" in text.lower()


class TestRangersCommand:
    @pytest.mark.asyncio
    async def test_rangers_lists_all(self):
        add_ranger(
            "Иванов Иван",
            chat_id=2300,
            badge_number="001",
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        add_ranger(
            "Петров Пётр",
            chat_id=2301,
            badge_number="002",
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        # No ADMIN_CHAT_IDS set — anyone can use it
        ADMIN_CHAT_IDS.clear()
        update = _make_update(chat_id=2300)
        await rangers_cmd(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "Иванов" in text
        assert "Петров" in text
        assert "2" in text  # count

    @pytest.mark.asyncio
    async def test_rangers_admin_restricted(self):
        ADMIN_CHAT_IDS.add(9999)
        update = _make_update(chat_id=2400)
        await rangers_cmd(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "администратор" in text.lower()
        ADMIN_CHAT_IDS.discard(9999)

    @pytest.mark.asyncio
    async def test_rangers_empty(self):
        ADMIN_CHAT_IDS.clear()
        update = _make_update(chat_id=2500)
        await rangers_cmd(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "нет зарегистрированных" in text.lower()


class TestTextFallback:
    @pytest.mark.asyncio
    @patch("cloud.notify.bot_handlers.get_active_incident_for_chat", return_value=None)
    async def test_unregistered_user_gets_start_hint(self, _mock_incident):
        update = _make_update(chat_id=5000, text="привет")
        await text_handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "/start" in text

    @pytest.mark.asyncio
    @patch("cloud.notify.bot_handlers.get_active_incident_for_chat", return_value=None)
    async def test_registered_user_gets_help_hint(self, _mock_incident):
        add_ranger(
            "Фоллбэк Тест",
            chat_id=5100,
            zone_lat_min=57.0,
            zone_lat_max=58.0,
            zone_lon_min=44.0,
            zone_lon_max=46.0,
        )
        update = _make_update(chat_id=5100, text="привет")
        await text_handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "/help" in text


class TestSnoozeCallback:
    @pytest.mark.asyncio
    async def test_snooze_edits_message(self):
        from cloud.db.incidents import create_incident

        incident = create_incident(
            audio_class="chainsaw",
            lat=57.3,
            lon=44.8,
            confidence=0.85,
            gating_level="alert",
        )
        update = _make_callback_update(chat_id=2600, data=f"snooze:{incident.id}")
        ctx = MagicMock()
        ctx.job_queue = MagicMock()
        await snooze_callback(update, ctx)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "отложено" in text.lower()


class TestRegistrationProgress:
    @pytest.mark.asyncio
    async def test_step1_shows_progress(self):
        update = _make_callback_update(chat_id=2700, data="district:varnavino")
        await district_chosen(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "шаг 1 из 3" in text.lower()

    @pytest.mark.asyncio
    async def test_step2_shows_progress(self):
        _registration_state[2800] = {
            "step": _REG_STEP_NAME,
            "district_slug": "varnavino",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=2800, text="Иванов Иван Иванович")
        await text_handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "шаг 2 из 3" in text.lower()

    @pytest.mark.asyncio
    async def test_step3_shows_progress(self):
        _registration_state[2900] = {
            "step": _REG_STEP_BADGE,
            "district_slug": "varnavino",
            "name": "Иванов Иван Иванович",
            "started_at": time.time(),
        }
        update = _make_update(chat_id=2900, text="12345")
        await text_handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "шаг 3 из 3" in text.lower()
