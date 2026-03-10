"""Notification pipeline tests — anti-spam, zone routing, rate limiting.

Demonstrates every protection layer that prevents the Telegram bot
from spamming rangers:

  1. Zone routing: alerts only reach rangers whose zone covers the event
  2. No fallback: uncovered areas produce NO messages (no admin spam)
  3. Rate limiting: max 1 alert per 5 minutes per ranger
  4. Permit suppression: legal forestry work is silently ignored
  5. Full scenarios: end-to-end from audio detection to Telegram delivery
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from cloud.agent.decision import Alert
from cloud.db.rangers import (
    add_ranger,
    get_rangers_for_location,
    init_db as init_rangers_db,
)
from cloud.db.permits import (
    add_permit,
    init_db as init_permits_db,
)
from cloud.notify import telegram
from cloud.notify.telegram import (
    send_pending,
    send_confirmed,
    _get_target_chat_ids,
    _is_rate_limited,
    _mark_sent,
    _last_sent,
    COOLDOWN_SECONDS,
)
from cloud.notify.districts import DISTRICTS
from edge.audio.classifier import AudioResult
from edge.audio.onset import OnsetDetector
from edge.decision.decider import decide
from edge.tdoa.triangulate import TriangulationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = date.today()
NEXT_MONTH = TODAY + timedelta(days=30)

# Varnavino district center (Нижегородская область)
VARNAVINO_LAT, VARNAVINO_LON = 57.30, 45.00

# Moscow coordinates (NOT covered by any district)
MOSCOW_LAT, MOSCOW_LON = 55.75, 37.61


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path, monkeypatch):
    """Each test gets fresh databases and cleared rate limiter."""
    monkeypatch.setenv("RANGERS_DB_PATH", str(tmp_path / "rangers.sqlite"))
    monkeypatch.setenv("PERMITS_DB_PATH", str(tmp_path / "permits.sqlite"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    init_rangers_db()
    init_permits_db()
    _last_sent.clear()


def _register_varnavino_ranger(name: str, chat_id: int):
    """Register a ranger for Varnavino district."""
    d = DISTRICTS["varnavino"]
    return add_ranger(
        name,
        chat_id,
        zone_lat_min=d.lat_min,
        zone_lat_max=d.lat_max,
        zone_lon_min=d.lon_min,
        zone_lon_max=d.lon_max,
    )


def _audio(label: str, confidence: float = 0.92) -> AudioResult:
    scores = {
        k: 0.0 for k in ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]
    }
    if label in scores:
        scores[label] = confidence
    return AudioResult(label=label, confidence=confidence, raw_scores=scores)


def _location(lat: float, lon: float) -> TriangulationResult:
    return TriangulationResult(lat=lat, lon=lon, error_m=5.0)


def _alert(lat: float, lon: float, priority: str = "ВЫСОКИЙ") -> Alert:
    return Alert(
        text="Обнаружена незаконная вырубка леса",
        priority=priority,
        lat=lat,
        lon=lon,
    )


# ===================================================================
# 1. ZONE-BASED ROUTING
# ===================================================================


class TestZoneRouting:
    """Alerts reach ONLY rangers whose zone covers the detection point."""

    def test_ranger_in_zone_receives_alert(self):
        """Chainsaw in Varnavino -> ranger in Varnavino gets notified."""
        _register_varnavino_ranger("Лесник Василий", 1001)

        targets = _get_target_chat_ids(VARNAVINO_LAT, VARNAVINO_LON)
        assert targets == [1001]

    def test_ranger_outside_zone_not_notified(self):
        """Chainsaw in Moscow -> Varnavino ranger does NOT get notified."""
        _register_varnavino_ranger("Лесник Василий", 1001)

        targets = _get_target_chat_ids(MOSCOW_LAT, MOSCOW_LON)
        assert targets == []

    def test_multiple_rangers_all_receive(self):
        """Two rangers in Varnavino -> both get the alert."""
        _register_varnavino_ranger("Иванов", 2001)
        _register_varnavino_ranger("Петров", 2002)

        targets = _get_target_chat_ids(VARNAVINO_LAT, VARNAVINO_LON)
        assert sorted(targets) == [2001, 2002]

    def test_no_rangers_means_no_targets(self):
        """No rangers registered at all -> empty target list."""
        targets = _get_target_chat_ids(VARNAVINO_LAT, VARNAVINO_LON)
        assert targets == []

    def test_no_fallback_to_admin_chat(self):
        """Even with TELEGRAM_CHAT_ID set, uncovered areas get NO message."""
        _register_varnavino_ranger("Лесник", 3001)

        # Detection in Moscow -- no ranger covers it
        targets = _get_target_chat_ids(MOSCOW_LAT, MOSCOW_LON)
        assert targets == [], "Must NOT fall back to admin chat"


# ===================================================================
# 2. RATE LIMITING
# ===================================================================


class TestRateLimiting:
    """Each ranger gets at most 1 alert per COOLDOWN_SECONDS."""

    def test_first_alert_not_rate_limited(self):
        """First alert to a ranger always goes through."""
        assert not _is_rate_limited(5001)

    def test_second_alert_within_cooldown_blocked(self):
        """Immediate second alert is blocked by rate limiter."""
        _mark_sent(5002)
        assert _is_rate_limited(5002)

    def test_alert_after_cooldown_allowed(self, monkeypatch):
        """After cooldown expires, ranger can receive alerts again."""
        _mark_sent(5003)
        # Simulate time passing beyond cooldown
        _last_sent[5003] = time.monotonic() - COOLDOWN_SECONDS - 1
        assert not _is_rate_limited(5003)

    def test_different_rangers_independent_limits(self):
        """Rate limit is per-ranger, not global."""
        _mark_sent(6001)
        assert _is_rate_limited(6001)
        assert not _is_rate_limited(6002)  # different ranger

    def test_cooldown_default_is_five_minutes(self):
        """Default cooldown is 300 seconds (5 minutes)."""
        assert COOLDOWN_SECONDS == 300


# ===================================================================
# 3. TELEGRAM DELIVERY (with mocked Bot)
# ===================================================================


class TestTelegramDelivery:
    """Verify actual Telegram calls respect zone + rate limit."""

    @pytest.fixture()
    def mock_bot(self, monkeypatch):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        bot.send_photo = AsyncMock()
        monkeypatch.setattr("cloud.notify.telegram.Bot", lambda token: bot)
        return bot

    @pytest.mark.asyncio
    async def test_pending_alert_sent_to_zone_ranger(self, mock_bot):
        """Pending alert reaches the ranger covering detection zone."""
        _register_varnavino_ranger("Алексей", 7001)

        await send_pending(VARNAVINO_LAT, VARNAVINO_LON, "chainsaw", "illegal")

        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args
        assert call_kwargs.kwargs["chat_id"] == 7001
        assert "Бензопила" in call_kwargs.kwargs["text"]

    @pytest.mark.asyncio
    async def test_pending_alert_silent_outside_zone(self, mock_bot):
        """Pending alert for uncovered area -> NO Telegram call."""
        _register_varnavino_ranger("Алексей", 7001)

        await send_pending(MOSCOW_LAT, MOSCOW_LON, "chainsaw", "illegal")

        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirmed_alert_with_photo(self, mock_bot):
        """Confirmed alert with drone photo sends via send_photo."""
        _register_varnavino_ranger("Фотограф", 7002)

        alert = _alert(VARNAVINO_LAT, VARNAVINO_LON)
        await send_confirmed(alert, b"fake-photo-data")

        mock_bot.send_photo.assert_called_once()
        assert mock_bot.send_photo.call_args.kwargs["chat_id"] == 7002

    @pytest.mark.asyncio
    async def test_confirmed_alert_without_photo(self, mock_bot):
        """Confirmed alert without photo falls back to text message."""
        _register_varnavino_ranger("Текстовик", 7003)

        alert = _alert(VARNAVINO_LAT, VARNAVINO_LON)
        await send_confirmed(alert, None)

        mock_bot.send_message.assert_called_once()
        mock_bot.send_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limited_alert_not_sent(self, mock_bot):
        """Second rapid alert to same ranger is silently dropped."""
        _register_varnavino_ranger("Спамер", 7004)

        await send_pending(VARNAVINO_LAT, VARNAVINO_LON, "chainsaw", "x")
        assert mock_bot.send_message.call_count == 1

        # Second call within cooldown -> blocked
        await send_pending(VARNAVINO_LAT, VARNAVINO_LON, "gunshot", "y")
        assert mock_bot.send_message.call_count == 1  # still 1, not 2

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_active_rangers(self, mock_bot):
        """broadcast=True sends alert to ALL active rangers, ignoring zone."""
        _register_varnavino_ranger("Варнавино", 8501)
        # Moscow ranger — zone doesn't cover Varnavino, but broadcast ignores zone
        from cloud.db.rangers import add_ranger

        add_ranger(
            "Москва",
            8502,
            zone_lat_min=55.0,
            zone_lat_max=56.0,
            zone_lon_min=37.0,
            zone_lon_max=38.0,
        )

        await send_pending(
            MOSCOW_LAT,
            MOSCOW_LON,
            "chainsaw",
            "drone",
            confidence=0.85,
            broadcast=True,
        )

        assert mock_bot.send_message.call_count == 2
        sent_ids = {
            call.kwargs["chat_id"] for call in mock_bot.send_message.call_args_list
        }
        assert sent_ids == {8501, 8502}

    @pytest.mark.asyncio
    async def test_broadcast_skips_inactive_rangers(self, mock_bot):
        """broadcast=True does NOT send to inactive rangers."""
        _register_varnavino_ranger("Активный", 8601)
        from cloud.db.rangers import add_ranger, set_active

        add_ranger(
            "Неактивный",
            8602,
            zone_lat_min=55.0,
            zone_lat_max=56.0,
            zone_lon_min=37.0,
            zone_lon_max=38.0,
        )
        set_active(8602, False)

        await send_pending(
            MOSCOW_LAT,
            MOSCOW_LON,
            "chainsaw",
            "drone",
            confidence=0.85,
            broadcast=True,
        )

        assert mock_bot.send_message.call_count == 1
        assert mock_bot.send_message.call_args.kwargs["chat_id"] == 8601

    @pytest.mark.asyncio
    async def test_two_rangers_one_rate_limited(self, mock_bot):
        """Two rangers in zone: one is rate-limited, other still receives."""
        _register_varnavino_ranger("Быстрый", 8001)
        _register_varnavino_ranger("Новый", 8002)

        # First alert -> both receive
        await send_pending(VARNAVINO_LAT, VARNAVINO_LON, "chainsaw", "x")
        assert mock_bot.send_message.call_count == 2

        # Reset rate limit for ranger 8002 only
        _last_sent[8002] = time.monotonic() - COOLDOWN_SECONDS - 1

        # Second alert -> only 8002 receives (8001 still rate-limited)
        await send_pending(VARNAVINO_LAT, VARNAVINO_LON, "gunshot", "y")
        assert mock_bot.send_message.call_count == 3  # 2 + 1


# ===================================================================
# 4. FULL SCENARIO TESTS
# ===================================================================


class TestScenarioIllegalLogging:
    """Scenario: illegal chainsaw in Varnavino forest, no permit."""

    def test_onset_detects_chainsaw(self):
        """Chainsaw audio triggers onset detector (sharp transient)."""
        import numpy as np

        detector = OnsetDetector()
        # Simulate chainsaw burst: sudden loud signal
        silence = np.zeros(8000, dtype=np.float32)
        burst = np.random.RandomState(42).randn(8000).astype(np.float32) * 0.8
        waveform = np.concatenate([silence, burst])

        event = detector.detect(waveform, 16000)
        assert event.triggered, (
            f"Chainsaw should trigger onset (ratio={event.energy_ratio:.1f})"
        )

    def test_decider_sends_drone(self):
        """Chainsaw at 92% confidence, no permit -> send drone."""
        decision = decide(
            _audio("chainsaw", 0.92), _location(VARNAVINO_LAT, VARNAVINO_LON)
        )
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_alert_reaches_ranger(self):
        """Ranger registered in Varnavino receives chainsaw alert."""
        _register_varnavino_ranger("Егерь Николай", 9001)
        decision = decide(_audio("chainsaw"), _location(VARNAVINO_LAT, VARNAVINO_LON))
        rangers = get_rangers_for_location(VARNAVINO_LAT, VARNAVINO_LON)

        assert decision.send_drone is True
        assert len(rangers) == 1
        assert rangers[0].chat_id == 9001


class TestScenarioLegalForestry:
    """Scenario: chainsaw with valid logging permit (санитарная рубка)."""

    def test_permit_suppresses_alert(self):
        """Valid permit -> decider says NO drone, NO alert."""
        add_permit(57.0, 57.5, 44.5, 45.5, TODAY, NEXT_MONTH, "Санитарная рубка")
        decision = decide(_audio("chainsaw"), _location(VARNAVINO_LAT, VARNAVINO_LON))

        assert decision.send_drone is False
        assert decision.send_lora is False
        assert "permit" in decision.reason.lower()

    def test_ranger_not_disturbed(self):
        """Ranger exists but pipeline stops at decider -> no notification."""
        _register_varnavino_ranger("Тихий лесник", 9002)
        add_permit(57.0, 57.5, 44.5, 45.5, TODAY, NEXT_MONTH, "Рубка")

        decision = decide(_audio("chainsaw"), _location(VARNAVINO_LAT, VARNAVINO_LON))
        rangers = get_rangers_for_location(VARNAVINO_LAT, VARNAVINO_LON)

        assert decision.send_drone is False  # pipeline stops here
        assert len(rangers) == 1  # ranger exists but won't be bothered


class TestScenarioGunshot:
    """Scenario: gunshot detected -- ALWAYS alert, permits don't help."""

    def test_gunshot_ignores_permit(self):
        """Gunshot in zone with logging permit -> STILL alerts."""
        add_permit(57.0, 57.5, 44.5, 45.5, TODAY, NEXT_MONTH, "Рубка")
        decision = decide(
            _audio("gunshot", 0.95), _location(VARNAVINO_LAT, VARNAVINO_LON)
        )

        assert decision.send_drone is True
        assert decision.priority == "high"

    def test_fire_ignores_permit(self):
        """Fire in zone with logging permit -> STILL alerts."""
        add_permit(57.0, 57.5, 44.5, 45.5, TODAY, NEXT_MONTH, "Рубка")
        decision = decide(_audio("fire", 0.88), _location(VARNAVINO_LAT, VARNAVINO_LON))

        assert decision.send_drone is True
        assert decision.priority == "high"


class TestScenarioFalsePositive:
    """Scenario: background noise or low-confidence detection."""

    def test_background_noise_ignored(self):
        """Background audio -> no drone, no alert."""
        decision = decide(
            _audio("background", 0.95), _location(VARNAVINO_LAT, VARNAVINO_LON)
        )
        assert decision.send_drone is False

    def test_low_confidence_ignored(self):
        """Chainsaw at 40% confidence -> too uncertain, no alert."""
        decision = decide(
            _audio("chainsaw", 0.40), _location(VARNAVINO_LAT, VARNAVINO_LON)
        )
        assert decision.send_drone is False
        assert "confidence" in decision.reason.lower()

    def test_detection_outside_all_zones_silent(self):
        """Chainsaw in Moscow (no rangers there) -> no one to notify."""
        _register_varnavino_ranger("Далёкий лесник", 9003)
        targets = _get_target_chat_ids(MOSCOW_LAT, MOSCOW_LON)
        assert targets == []


class TestScenarioExpiredPermit:
    """Scenario: permit expired yesterday, chainsaw today."""

    def test_expired_permit_does_not_suppress(self):
        """Expired permit -> chainsaw triggers full alert."""
        yesterday = TODAY - timedelta(days=1)
        last_year = TODAY - timedelta(days=365)
        add_permit(57.0, 57.5, 44.5, 45.5, last_year, yesterday, "Истёкший")

        decision = decide(_audio("chainsaw"), _location(VARNAVINO_LAT, VARNAVINO_LON))
        assert decision.send_drone is True
        assert decision.priority == "high"
