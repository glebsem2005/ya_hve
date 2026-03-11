"""Tests for edge.decision.decider — the decision engine.

Covers priority mapping, confidence thresholds, safe-class bypass,
logging permit suppression, and output dataclass structure.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from edge.audio.classifier import AudioResult
from edge.decision.decider import (
    ALERT_THRESHOLDS,
    CONFIDENCE_ALERT,
    CONFIDENCE_VERIFY,
    Decision,
    PERMIT_CLASSES,
    PRIORITY_MAP,
    SAFE_CLASSES,
    decide,
)
from edge.tdoa.triangulate import TriangulationResult
from cloud.db.permits import add_permit, init_db as init_permits_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ar(label: str, confidence: float) -> AudioResult:
    """Shorthand to build an AudioResult for tests."""
    scores = {
        "chainsaw": 0.0,
        "gunshot": 0.0,
        "engine": 0.0,
        "axe": 0.0,
        "fire": 0.0,
        "background": 0.0,
    }
    if label in scores:
        scores[label] = confidence
    return AudioResult(label=label, confidence=confidence, raw_scores=scores)


def _loc(lat: float = 55.7510, lon: float = 37.6130) -> TriangulationResult:
    return TriangulationResult(lat=lat, lon=lon, error_m=5.0)


@pytest.fixture(autouse=True)
def _fresh_permits_db(tmp_path, monkeypatch):
    """Each test gets a fresh permits database (empty = no permits)."""
    db_file = str(tmp_path / "permits_test.sqlite")
    monkeypatch.setenv("PERMITS_DB_PATH", db_file)
    init_permits_db()


# ---------------------------------------------------------------------------
# Safe-class bypass
# ---------------------------------------------------------------------------


class TestSafeClass:
    def test_safe_class_background_no_drone(self) -> None:
        decision = decide(_ar("background", 0.99), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False
        assert decision.priority == "low"

    def test_safe_classes_only_background(self) -> None:
        assert SAFE_CLASSES == {"background"}

    def test_no_silence_in_safe_classes(self) -> None:
        assert "silence" not in SAFE_CLASSES


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_log_only_below_verify(self) -> None:
        """Below CONFIDENCE_VERIFY → log only, no action."""
        decision = decide(_ar("chainsaw", 0.30), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False
        assert decision.priority == "low"

    def test_verify_zone_no_drone(self) -> None:
        """Between VERIFY and ALERT → LoRa only, no drone."""
        decision = decide(_ar("chainsaw", 0.50), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_confidence_at_alert_threshold(self) -> None:
        """At CONFIDENCE_ALERT → full alert (drone + LoRa)."""
        decision = decide(_ar("chainsaw", CONFIDENCE_ALERT), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True

    def test_confidence_below_alert_threshold(self) -> None:
        """Just below CONFIDENCE_ALERT → verify zone (LoRa only)."""
        decision = decide(_ar("chainsaw", 0.69), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True


# ---------------------------------------------------------------------------
# Differentiated alert thresholds per class
# ---------------------------------------------------------------------------


class TestDifferentiatedThresholds:
    """Critical threats (gunshot, fire) have lower ALERT threshold (0.55).
    Other classes keep the default 0.70."""

    def test_alert_thresholds_dict_exists(self) -> None:
        assert isinstance(ALERT_THRESHOLDS, dict)
        assert "gunshot" in ALERT_THRESHOLDS
        assert "fire" in ALERT_THRESHOLDS

    def test_gunshot_threshold_is_055(self) -> None:
        assert ALERT_THRESHOLDS["gunshot"] == 0.55

    def test_fire_threshold_is_055(self) -> None:
        assert ALERT_THRESHOLDS["fire"] == 0.55

    def test_chainsaw_threshold_is_070(self) -> None:
        assert ALERT_THRESHOLDS["chainsaw"] == 0.70

    def test_gunshot_alert_at_055(self) -> None:
        """Gunshot at 0.55 → ALERT (drone flies)."""
        decision = decide(_ar("gunshot", 0.55), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_gunshot_verify_at_050(self) -> None:
        """Gunshot at 0.50 → VERIFY (no drone, LoRa only)."""
        decision = decide(_ar("gunshot", 0.50), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True

    def test_fire_alert_at_055(self) -> None:
        """Fire at 0.55 → ALERT (drone flies)."""
        decision = decide(_ar("fire", 0.55), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_fire_verify_at_050(self) -> None:
        """Fire at 0.50 → VERIFY (no drone)."""
        decision = decide(_ar("fire", 0.50), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True

    def test_chainsaw_still_needs_070(self) -> None:
        """Chainsaw at 0.65 → VERIFY (threshold unchanged at 0.70)."""
        decision = decide(_ar("chainsaw", 0.65), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True

    def test_chainsaw_at_070_alerts(self) -> None:
        """Chainsaw at 0.70 → ALERT (as before)."""
        decision = decide(_ar("chainsaw", 0.70), _loc())
        assert decision.send_drone is True

    def test_engine_still_needs_070(self) -> None:
        """Engine at 0.65 → VERIFY (threshold unchanged)."""
        decision = decide(_ar("engine", 0.65), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True

    def test_axe_still_needs_070(self) -> None:
        """Axe at 0.65 → VERIFY."""
        decision = decide(_ar("axe", 0.65), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is True


# ---------------------------------------------------------------------------
# Priority mapping — high
# ---------------------------------------------------------------------------


class TestHighPriority:
    @pytest.mark.parametrize("label", ["chainsaw", "gunshot", "fire", "axe"])
    def test_high_priority(self, label: str) -> None:
        decision = decide(_ar(label, 0.90), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_high_priority_chainsaw(self) -> None:
        decision = decide(_ar("chainsaw", 0.90), _loc())
        assert decision.priority == "high"

    def test_high_priority_gunshot(self) -> None:
        decision = decide(_ar("gunshot", 0.90), _loc())
        assert decision.priority == "high"

    def test_high_priority_fire(self) -> None:
        decision = decide(_ar("fire", 0.90), _loc())
        assert decision.priority == "high"

    def test_high_priority_axe(self) -> None:
        decision = decide(_ar("axe", 0.90), _loc())
        assert decision.priority == "high"


# ---------------------------------------------------------------------------
# Medium / low priorities
# ---------------------------------------------------------------------------


class TestOtherPriorities:
    def test_medium_priority_engine(self) -> None:
        decision = decide(_ar("engine", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.priority == "medium"

    def test_unknown_low_confidence(self) -> None:
        decision = decide(_ar("unknown", 0.50), _loc())
        assert decision.send_drone is False

    def test_unknown_high_confidence(self) -> None:
        decision = decide(_ar("unknown", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "low"


# ---------------------------------------------------------------------------
# PRIORITY_MAP structure
# ---------------------------------------------------------------------------


class TestPriorityMap:
    def test_priority_map_has_all_classes(self) -> None:
        expected_keys = {
            "chainsaw",
            "gunshot",
            "engine",
            "axe",
            "fire",
            "background",
            "unknown",
        }
        assert set(PRIORITY_MAP.keys()) == expected_keys

    def test_no_birds_in_priority_map(self) -> None:
        assert "birds" not in PRIORITY_MAP


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestDecisionOutput:
    def test_coordinates_in_reason(self) -> None:
        loc = _loc(lat=55.7510, lon=37.6130)
        decision = decide(_ar("chainsaw", 0.90), loc)
        assert "55.7510" in decision.reason
        assert "37.6130" in decision.reason

    def test_decision_dataclass_fields(self) -> None:
        d = Decision(send_drone=True, send_lora=True, priority="high", reason="test")
        assert hasattr(d, "send_drone")
        assert hasattr(d, "send_lora")
        assert hasattr(d, "priority")
        assert hasattr(d, "reason")


# ---------------------------------------------------------------------------
# Logging permit suppression (лесные билеты)
# ---------------------------------------------------------------------------

TODAY = date.today()
NEXT_MONTH = TODAY + timedelta(days=30)
YESTERDAY = TODAY - timedelta(days=1)
LAST_YEAR = TODAY - timedelta(days=365)

# Zone covering the default test location (55.751, 37.613)
PERMIT_ZONE = dict(
    zone_lat_min=55.0, zone_lat_max=56.0, zone_lon_min=37.0, zone_lon_max=38.0
)


class TestPermitSuppression:
    """Chainsaw/axe/engine with a valid permit → no alert.
    Without permit or for gunshot/fire → alert as usual."""

    def test_chainsaw_with_permit_no_alert(self) -> None:
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("chainsaw", 0.90), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False
        assert decision.priority == "low"
        assert "permit" in decision.reason.lower()

    def test_axe_with_permit_no_alert(self) -> None:
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("axe", 0.90), _loc())
        assert decision.send_drone is False
        assert decision.priority == "low"

    def test_engine_with_permit_no_alert(self) -> None:
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("engine", 0.85), _loc())
        assert decision.send_drone is False
        assert decision.priority == "low"

    def test_chainsaw_without_permit_alerts(self) -> None:
        """No permit → full alert."""
        decision = decide(_ar("chainsaw", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_chainsaw_expired_permit_alerts(self) -> None:
        """Expired permit → alert (illegal logging)."""
        add_permit(**PERMIT_ZONE, valid_from=LAST_YEAR, valid_until=YESTERDAY)
        decision = decide(_ar("chainsaw", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.priority == "high"

    def test_chainsaw_wrong_zone_alerts(self) -> None:
        """Permit in different zone → alert."""
        add_permit(
            zone_lat_min=60.0,
            zone_lat_max=61.0,
            zone_lon_min=30.0,
            zone_lon_max=31.0,
            valid_from=TODAY,
            valid_until=NEXT_MONTH,
        )
        decision = decide(_ar("chainsaw", 0.90), _loc())
        assert decision.send_drone is True

    def test_gunshot_with_permit_still_alerts(self) -> None:
        """Gunshot is NEVER covered by a logging permit."""
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("gunshot", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.priority == "high"

    def test_fire_with_permit_still_alerts(self) -> None:
        """Fire is NEVER covered by a logging permit."""
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("fire", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.priority == "high"

    def test_permit_classes_constant(self) -> None:
        """Only chainsaw, axe, engine are covered by permits."""
        assert PERMIT_CLASSES == {"chainsaw", "axe", "engine"}

    def test_low_confidence_skips_permit_check(self) -> None:
        """Below CONFIDENCE_VERIFY exits before permit check."""
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("chainsaw", 0.30), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False

    def test_background_skips_permit_check(self) -> None:
        """Background exits before permit check."""
        add_permit(**PERMIT_ZONE, valid_from=TODAY, valid_until=NEXT_MONTH)
        decision = decide(_ar("background", 0.99), _loc())
        assert decision.send_drone is False
        assert "safe" in decision.reason.lower()
