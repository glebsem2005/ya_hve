"""Tests: demo alerts reach Telegram via broadcast=True."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _real_main():
    """Ensure cloud.interface.main is the real module, not a MagicMock."""
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F811

    return sys.modules[mod_name]


def _make_deps(*, send_drone: bool = True, send_lora: bool = False):
    """Return a minimal deps dict with configurable decision flags."""
    MicPosition = types.SimpleNamespace
    mic_sim = AsyncMock()
    mic_sim.get_signals.return_value = (
        [[0.0] * 16000, [0.0] * 16000, [0.0] * 16000],
        ["/tmp/a.wav", "/tmp/b.wav", "/tmp/c.wav"],
    )

    onset = types.SimpleNamespace(triggered=True, energy_ratio=5.0)
    audio_result = types.SimpleNamespace(
        label="chainsaw",
        confidence=0.92,
        raw_scores={"chainsaw": 0.92, "background": 0.08},
    )
    location = types.SimpleNamespace(lat=57.37, lon=44.63, error_m=50.0)
    decision = types.SimpleNamespace(
        send_drone=send_drone,
        send_lora=send_lora,
        priority="P0",
        reason="chainsaw detected",
    )

    photo = types.SimpleNamespace(b64="AAAA", data=b"\x00")
    drone = AsyncMock()
    drone.fly_to.return_value = AsyncMock(__aiter__=AsyncMock(return_value=iter([])))
    drone.capture_photo.return_value = photo

    incident = types.SimpleNamespace(id=42)
    vision = types.SimpleNamespace(
        description="trees",
        has_human=False,
        has_fire=False,
        has_felling=True,
        has_machinery=False,
    )

    return {
        "MicPosition": MicPosition,
        "MicSimulator": MagicMock(return_value=mic_sim),
        "detect_onset": MagicMock(return_value=onset),
        "classify": MagicMock(return_value=audio_result),
        "triangulate": MagicMock(return_value=location),
        "decide": MagicMock(return_value=decision),
        "SimulatedDrone": MagicMock(return_value=drone),
        "classify_photo": AsyncMock(return_value=vision),
        "compose_alert": AsyncMock(),
        "send_pending": AsyncMock(return_value=incident),
        "send_confirmed": AsyncMock(),
        "get_online": MagicMock(return_value=[]),
    }


# ---------- Test 1: alert/drone path sends broadcast=True ----------


@pytest.mark.asyncio
async def test_demo_send_pending_uses_broadcast(_real_main):
    """When send_drone=True, send_pending must be called with broadcast=True."""
    deps = _make_deps(send_drone=True)
    _run_demo = _real_main._run_demo

    with (
        patch.object(_real_main, "_import_demo_deps", return_value=deps),
        patch.object(_real_main, "broadcast", new_callable=AsyncMock),
        patch(
            "cloud.db.microphones.random_point_in_boundary",
            return_value=(57.37, 44.63),
        ),
        patch(
            "cloud.db.microphones.get_nearest_online",
            return_value=[],
        ),
    ):
        await _run_demo("chainsaw")

    deps["send_pending"].assert_called_once()
    _, kwargs = deps["send_pending"].call_args
    assert kwargs.get("broadcast") is True, (
        f"send_pending must use broadcast=True for demo, got {kwargs}"
    )


# ---------- Test 2: verify path sends broadcast=True ----------


@pytest.mark.asyncio
async def test_demo_verify_send_pending_uses_broadcast(_real_main):
    """When send_drone=False but send_lora=True, send_pending must use broadcast=True."""
    deps = _make_deps(send_drone=False, send_lora=True)
    _run_demo = _real_main._run_demo

    with (
        patch.object(_real_main, "_import_demo_deps", return_value=deps),
        patch.object(_real_main, "broadcast", new_callable=AsyncMock),
        patch(
            "cloud.db.microphones.random_point_in_boundary",
            return_value=(57.37, 44.63),
        ),
        patch(
            "cloud.db.microphones.get_nearest_online",
            return_value=[],
        ),
    ):
        await _run_demo("chainsaw")

    deps["send_pending"].assert_called_once()
    _, kwargs = deps["send_pending"].call_args
    assert kwargs.get("broadcast") is True, (
        f"send_pending must use broadcast=True for demo verify path, got {kwargs}"
    )


# ---------- Test 3: broadcast=True skips spatial dedup ----------


@pytest.mark.asyncio
async def test_demo_skips_spatial_dedup():
    """When broadcast=True, spatial dedup must not block the alert."""
    existing_incident = types.SimpleNamespace(id=99)

    with patch(
        "cloud.notify.telegram.get_recent_nearby_incident",
        return_value=existing_incident,
    ) as mock_dedup:
        from cloud.notify.telegram import send_pending

        # Patch remaining dependencies so send_pending doesn't hit real services
        with (
            patch("cloud.notify.telegram.Bot") as mock_bot_cls,
            patch("cloud.notify.telegram.get_all_rangers", return_value=[]),
            patch("cloud.notify.telegram.create_incident") as mock_create,
            patch("cloud.notify.telegram.get_nearest_rangers", return_value=[]),
        ):
            mock_create.return_value = types.SimpleNamespace(id=100)

            result = await send_pending(
                lat=57.37,
                lon=44.63,
                audio_class="chainsaw",
                reason="test",
                confidence=0.92,
                is_demo=True,
                broadcast=True,
            )

        # With broadcast=True, dedup should be skipped → new incident created
        assert result.id == 100, (
            f"Expected new incident (id=100), got id={result.id} — dedup was not skipped"
        )
