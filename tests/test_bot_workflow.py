"""Tests for the full inspector workflow: accept -> location -> verdict -> evidence -> PDF.

All external services (STT, Vision, RAG, YandexGPT) are mocked.
"""

import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# cloud.interface.main requires FastAPI (not in test env) — register mock module
import cloud.interface

_mock_main_module = MagicMock()
_mock_main_module.broadcast = AsyncMock()
sys.modules["cloud.interface.main"] = _mock_main_module
cloud.interface.main = _mock_main_module

_tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
_tmp.close()
os.environ["RANGERS_DB_PATH"] = _tmp.name

from cloud.db.rangers import init_db, add_ranger, get_ranger_by_chat_id
from cloud.db.incidents import (
    Incident,
    _incidents,
    _chat_to_incident,
    create_incident,
    get_incident,
    assign_chat_to_incident,
    update_incident,
)
from cloud.notify.bot_handlers import (
    accept_callback,
    location_handler,
    verdict_callback,
    voice_handler,
    handle_inspector_photo,
    text_handler,
    rag_callback,
    _generate_and_send_protocol,
    _registration_state,
)
from cloud.notify.telegram import CLASS_NAME_RU


# ---------- Helpers ----------


def _make_callback_update(chat_id: int, data: str, full_name: str = "Тест Тестов"):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message.chat_id = chat_id
    update.callback_query.message.reply_text = AsyncMock()
    update.callback_query.data = data
    update.callback_query.from_user.full_name = full_name
    update.callback_query.from_user.username = "test_user"
    return update


def _make_text_update(chat_id: int, text: str):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    update.callback_query = None
    return update


def _make_location_update(chat_id: int, lat: float, lon: float):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.location.latitude = lat
    update.message.location.longitude = lon
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    update.callback_query = None
    return update


def _make_voice_update(chat_id: int):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    update.message.voice.get_file = AsyncMock()
    update.message.voice.get_file.return_value.download_as_bytearray = AsyncMock(
        return_value=bytearray(b"fake-audio")
    )
    update.callback_query = None
    return update


def _make_photo_update(chat_id: int, caption: str | None = None):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    update.message.caption = caption
    mock_photo = MagicMock()
    mock_photo.get_file = AsyncMock()
    mock_photo.get_file.return_value.download_as_bytearray = AsyncMock(
        return_value=bytearray(b"fake-photo")
    )
    update.message.photo = [mock_photo]
    update.callback_query = None
    return update


def _create_test_incident(
    status: str = "pending",
    chat_id: int | None = None,
    audio_class: str = "chainsaw",
    lat: float = 57.3,
    lon: float = 45.0,
    confidence: float = 0.85,
) -> Incident:
    incident = create_incident(
        audio_class=audio_class,
        lat=lat,
        lon=lon,
        confidence=confidence,
        gating_level="alert",
    )
    incident.status = status
    if chat_id:
        incident.accepted_by_chat_id = chat_id
        incident.accepted_by_name = "Тест Тестов"
        incident.accepted_at = time.time()
        assign_chat_to_incident(chat_id, incident.id)
    return incident


# ---------- Fixtures ----------


@pytest.fixture(autouse=True)
def _fresh_state():
    import sqlite3

    conn = sqlite3.connect(os.environ["RANGERS_DB_PATH"])
    conn.execute("DROP TABLE IF EXISTS rangers")
    conn.commit()
    conn.close()
    init_db()
    _incidents.clear()
    _chat_to_incident.clear()
    _registration_state.clear()
    _mock_main_module.broadcast.reset_mock()
    yield


@pytest.fixture
def mock_bot():
    with patch("cloud.notify.bot_handlers.BOT_TOKEN", "fake-token"):
        with patch("cloud.notify.telegram.BOT_TOKEN", "fake-token"):
            with patch("telegram.Bot") as MockBot:
                bot_instance = AsyncMock()
                MockBot.return_value = bot_instance
                yield bot_instance


# ---------- TestAcceptCallback ----------


class TestAcceptCallback:
    @pytest.mark.asyncio
    async def test_accept_sets_status_accepted(self, mock_bot):
        add_ranger("Иванов Иван", chat_id=100, badge_number="111")
        incident = _create_test_incident(status="pending")
        incident.alert_message_ids = {100: 1}

        update = _make_callback_update(100, f"accept:{incident.id}")
        await accept_callback(update, MagicMock())

        assert incident.status == "accepted"
        assert incident.accepted_by_name == "Иванов Иван"  # from DB, not user.full_name
        assert incident.accepted_by_chat_id == 100

    @pytest.mark.asyncio
    async def test_accept_already_accepted(self, mock_bot):
        incident = _create_test_incident(status="accepted")
        incident.accepted_by_name = "Другой Инспектор"

        update = _make_callback_update(200, f"accept:{incident.id}")
        await accept_callback(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "уже принят" in text.lower() or "Другой Инспектор" in text

    @pytest.mark.asyncio
    async def test_accept_unknown_incident(self, mock_bot):
        update = _make_callback_update(300, "accept:nonexistent-id")
        await accept_callback(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "не найден" in text.lower()

    @pytest.mark.asyncio
    async def test_accept_edits_other_rangers_alerts(self, mock_bot):
        add_ranger("Инспектор А", chat_id=400, badge_number="AAA")
        add_ranger("Инспектор Б", chat_id=401, badge_number="BBB")
        incident = _create_test_incident(status="pending")
        incident.alert_message_ids = {400: 10, 401: 20}

        update = _make_callback_update(
            400, f"accept:{incident.id}", full_name="Инспектор А"
        )
        await accept_callback(update, MagicMock())

        # Bot should have tried to edit message for chat 401
        assert mock_bot.edit_message_text.called

    @pytest.mark.asyncio
    async def test_accept_sends_drone_photo(self, mock_bot):
        add_ranger("Тестер", chat_id=500, badge_number="555")
        incident = _create_test_incident(status="pending")
        incident.drone_photo_b64 = "ZmFrZQ=="  # base64 of "fake"
        incident.drone_comment = "Дрон видит рубку"
        incident.alert_message_ids = {500: 1}

        update = _make_callback_update(500, f"accept:{incident.id}")
        await accept_callback(update, MagicMock())

        # send_drone_photo uses Bot internally
        assert mock_bot.send_photo.called or mock_bot.send_message.called


# ---------- TestLocationHandler ----------


class TestLocationHandler:
    @pytest.mark.asyncio
    async def test_location_near_triggers_on_site(self, mock_bot):
        incident = _create_test_incident(status="accepted", chat_id=600)
        # Send location very close to incident
        update = _make_location_update(600, incident.lat, incident.lon)
        await location_handler(update, MagicMock())

        assert incident.status == "on_site"

    @pytest.mark.asyncio
    async def test_location_far_shows_distance(self):
        incident = _create_test_incident(status="accepted", chat_id=700)
        # Send location far from incident (~111 km away)
        update = _make_location_update(700, incident.lat + 1.0, incident.lon)
        await location_handler(update, MagicMock())

        assert incident.status == "accepted"  # unchanged
        text = update.message.reply_text.call_args[0][0]
        assert "м от точки" in text

    @pytest.mark.asyncio
    async def test_location_no_active_incident(self):
        update = _make_location_update(800, 57.0, 45.0)
        await location_handler(update, MagicMock())
        # Should not crash, just return
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_location_wrong_status_ignored(self):
        incident = _create_test_incident(status="on_site", chat_id=900)
        update = _make_location_update(900, incident.lat, incident.lon)
        await location_handler(update, MagicMock())
        # on_site is not in ("accepted",) so should be ignored
        update.message.reply_text.assert_not_called()


# ---------- TestVerdictCallback ----------


class TestVerdictCallback:
    @pytest.mark.asyncio
    async def test_confirmed_requests_evidence(self, mock_bot):
        incident = _create_test_incident(status="on_site", chat_id=1000)

        update = _make_callback_update(1000, f"verdict:confirmed:{incident.id}")
        await verdict_callback(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "зафиксировано" in text.lower()

    @pytest.mark.asyncio
    async def test_false_alarm_closes_incident(self):
        incident = _create_test_incident(status="on_site", chat_id=1100)

        update = _make_callback_update(1100, f"verdict:false:{incident.id}")
        await verdict_callback(update, MagicMock())

        assert incident.status == "false_alarm"
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "закрыт" in text.lower()

    @pytest.mark.asyncio
    async def test_unknown_incident_error(self):
        update = _make_callback_update(1200, "verdict:confirmed:nonexistent")
        await verdict_callback(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "не найден" in text.lower()

    @pytest.mark.asyncio
    async def test_malformed_data_no_crash(self):
        update = _make_callback_update(1300, "verdict:bad")
        await verdict_callback(update, MagicMock())
        # Should not crash — parts < 3 → early return


# ---------- TestVoiceHandler ----------


class TestVoiceHandler:
    @pytest.mark.asyncio
    @patch(
        "cloud.agent.stt.recognize_voice",
        new_callable=AsyncMock,
        return_value="Вижу бензопилу у дороги",
    )
    async def test_voice_recognized_no_photo(self, mock_stt, mock_bot):
        incident = _create_test_incident(status="on_site", chat_id=1400)

        update = _make_voice_update(1400)
        await voice_handler(update, MagicMock())

        assert incident.ranger_report_raw == "Вижу бензопилу у дороги"
        # No photo yet — should not generate protocol
        assert incident.protocol_pdf is None

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.stt.recognize_voice",
        new_callable=AsyncMock,
        return_value="Рубка леса",
    )
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        return_value="Юридический текст",
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol",
        new_callable=AsyncMock,
        return_value="Ст. 260 УК РФ",
    )
    @patch("cloud.agent.protocol_pdf.generate_protocol", return_value=b"fake-pdf")
    async def test_voice_plus_photo_triggers_protocol(
        self, mock_pdf, mock_rag, mock_legal, mock_stt, mock_bot
    ):
        incident = _create_test_incident(status="on_site", chat_id=1500)
        incident.ranger_photo_b64 = "ZmFrZQ=="  # photo already collected

        update = _make_voice_update(1500)
        await voice_handler(update, MagicMock())

        assert incident.ranger_report_raw == "Рубка леса"

    @pytest.mark.asyncio
    @patch("cloud.agent.stt.recognize_voice", new_callable=AsyncMock, return_value="")
    async def test_voice_recognition_failure_fallback(self, mock_stt):
        incident = _create_test_incident(status="on_site", chat_id=1600)

        update = _make_voice_update(1600)
        await voice_handler(update, MagicMock())

        # Empty recognition → fallback message
        text = update.message.reply_text.call_args[0][0]
        assert "текстом" in text.lower() or "распознать" in text.lower()

    @pytest.mark.asyncio
    async def test_voice_no_active_incident(self):
        update = _make_voice_update(1700)
        await voice_handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "нет активного" in text.lower()


# ---------- TestPhotoHandler ----------


class TestPhotoHandler:
    @pytest.mark.asyncio
    async def test_evidence_photo_on_site_saved(self):
        incident = _create_test_incident(status="on_site", chat_id=1800)

        update = _make_photo_update(1800)
        await handle_inspector_photo(update, MagicMock())

        assert incident.ranger_photo_b64 is not None
        # No report yet — should ask for description
        calls = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert any("описание" in t.lower() or "опишите" in t.lower() for t in calls)

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        return_value="Юридический текст",
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol",
        new_callable=AsyncMock,
        return_value="Ст. 260",
    )
    @patch("cloud.agent.protocol_pdf.generate_protocol", return_value=b"fake-pdf")
    async def test_photo_with_caption_triggers_protocol(
        self, mock_pdf, mock_rag, mock_legal, mock_bot
    ):
        incident = _create_test_incident(status="on_site", chat_id=1900)

        update = _make_photo_update(1900, caption="Вижу рубку леса")
        await handle_inspector_photo(update, MagicMock())

        assert incident.ranger_photo_b64 is not None
        assert incident.ranger_report_raw == "Вижу рубку леса"

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        return_value="Юр текст",
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol",
        new_callable=AsyncMock,
        return_value="Ст. 260",
    )
    @patch("cloud.agent.protocol_pdf.generate_protocol", return_value=b"fake-pdf")
    async def test_photo_with_existing_report_triggers_protocol(
        self, mock_pdf, mock_rag, mock_legal, mock_bot
    ):
        incident = _create_test_incident(status="on_site", chat_id=2000)
        incident.ranger_report_raw = "Ранее записанный отчёт"

        update = _make_photo_update(2000)
        await handle_inspector_photo(update, MagicMock())

        assert incident.ranger_photo_b64 is not None

    @pytest.mark.asyncio
    async def test_photo_no_incident_redirects_to_drone_bot(self):
        # No active incident — ranger bot redirects to drone bot
        update = _make_photo_update(2100)
        await handle_inspector_photo(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "нет активного инцидента" in text.lower()


# ---------- TestTextOnSite ----------


class TestTextOnSite:
    @pytest.mark.asyncio
    async def test_text_on_site_saves_report(self):
        incident = _create_test_incident(status="on_site", chat_id=2300)

        update = _make_text_update(2300, "Вижу рубку рядом с дорогой")
        await text_handler(update, MagicMock())

        assert incident.ranger_report_raw == "Вижу рубку рядом с дорогой"

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        return_value="Юр текст",
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol",
        new_callable=AsyncMock,
        return_value="Ст. 260",
    )
    @patch("cloud.agent.protocol_pdf.generate_protocol", return_value=b"fake-pdf")
    async def test_text_with_existing_photo_triggers_protocol(
        self, mock_pdf, mock_rag, mock_legal, mock_bot
    ):
        incident = _create_test_incident(status="on_site", chat_id=2400)
        incident.ranger_photo_b64 = "ZmFrZQ=="

        update = _make_text_update(2400, "Незаконная рубка подтверждена")
        await text_handler(update, MagicMock())

        assert incident.ranger_report_raw == "Незаконная рубка подтверждена"

    @pytest.mark.asyncio
    async def test_text_no_incident_ignored(self):
        update = _make_text_update(2500, "Просто текст")
        await text_handler(update, MagicMock())

        update.message.reply_text.assert_not_called()


# ---------- TestProtocolGeneration ----------


class TestProtocolGeneration:
    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        return_value="Юридическое описание",
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol",
        new_callable=AsyncMock,
        return_value="Ст. 260 УК РФ",
    )
    @patch("cloud.agent.protocol_pdf.generate_protocol", return_value=b"fake-pdf-bytes")
    async def test_full_protocol_flow(self, mock_pdf, mock_rag, mock_legal, mock_bot):
        incident = _create_test_incident(status="on_site", chat_id=2600)
        incident.ranger_report_raw = "Вижу рубку"
        incident.ranger_photo_b64 = "ZmFrZQ=="

        await _generate_and_send_protocol(2600, incident)

        assert incident.ranger_report_legal == "Юридическое описание"
        mock_pdf.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol", new_callable=AsyncMock, return_value=""
    )
    @patch("cloud.agent.protocol_pdf.generate_protocol", return_value=b"fake-pdf")
    async def test_legalize_failure_uses_raw(
        self, mock_pdf, mock_rag, mock_legal, mock_bot
    ):
        incident = _create_test_incident(status="on_site", chat_id=2700)
        incident.ranger_report_raw = "Рубка леса"

        await _generate_and_send_protocol(2700, incident)

        # Fallback: ranger_report_legal == ranger_report_raw
        assert incident.ranger_report_legal == "Рубка леса"

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent._call_yandex_plain",
        new_callable=AsyncMock,
        return_value="ok",
    )
    @patch(
        "cloud.agent.rag_agent.query_protocol", new_callable=AsyncMock, return_value=""
    )
    @patch(
        "cloud.agent.protocol_pdf.generate_protocol", side_effect=Exception("PDF error")
    )
    async def test_pdf_failure_sends_error(
        self, mock_pdf, mock_rag, mock_legal, mock_bot
    ):
        incident = _create_test_incident(status="on_site", chat_id=2800)
        incident.ranger_report_raw = "Отчёт"

        await _generate_and_send_protocol(2800, incident)

        # Should send error message
        assert mock_bot.send_message.called
        error_calls = [
            c
            for c in mock_bot.send_message.call_args_list
            if "ошибка" in str(c).lower() or "PDF" in str(c)
        ]
        assert len(error_calls) > 0


# ---------- TestRagCallback ----------


class TestRagCallback:
    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent.query_action",
        new_callable=AsyncMock,
        return_value="Вызовите наряд",
    )
    async def test_rag_action(self, mock_action):
        update = _make_callback_update(2900, "rag:action:chainsaw:57.3:45.0")
        await rag_callback(update, MagicMock())

        mock_action.assert_called_once_with("chainsaw", 57.3, 45.0)
        text = update.callback_query.message.reply_text.call_args[0][0]
        assert "Вызовите наряд" in text

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent.query_protocol",
        new_callable=AsyncMock,
        return_value="Ст. 260 УК РФ",
    )
    async def test_rag_protocol(self, mock_protocol):
        update = _make_callback_update(3000, "rag:protocol:gunshot:57.3:45.0")
        await rag_callback(update, MagicMock())

        mock_protocol.assert_called_once_with("gunshot", 57.3, 45.0)
        text = update.callback_query.message.reply_text.call_args[0][0]
        assert "260" in text

    @pytest.mark.asyncio
    @patch(
        "cloud.agent.rag_agent.query_action",
        new_callable=AsyncMock,
        side_effect=Exception("RAG failed"),
    )
    async def test_rag_failure_shows_error(self, mock_action):
        update = _make_callback_update(3100, "rag:action:chainsaw:57.3:45.0")
        await rag_callback(update, MagicMock())

        text = update.callback_query.message.reply_text.call_args[0][0]
        assert "ошибка" in text.lower() or "RAG" in text
