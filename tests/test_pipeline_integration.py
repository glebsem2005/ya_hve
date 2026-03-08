"""Integration tests — end-to-end pipeline: ranger registration → threat
detection → permit check → alert routing.

Verifies that:
1. Ranger registration connects to alert routing
2. Detected threats at a ranger's zone produce alerts
3. Valid logging permits suppress chainsaw/axe/engine alerts
4. Gunshot/fire always alert regardless of permits
5. Expired permits don't suppress alerts
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cloud.db.rangers import (
    add_ranger,
    get_rangers_for_location,
    init_db as init_rangers_db,
)
from cloud.db.permits import (
    add_permit,
    has_valid_permit,
    init_db as init_permits_db,
)
from cloud.notify.districts import DISTRICTS
from edge.audio.classifier import AudioResult
from edge.decision.decider import decide
from edge.tdoa.triangulate import TriangulationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = date.today()
NEXT_MONTH = TODAY + timedelta(days=30)
YESTERDAY = TODAY - timedelta(days=1)
LAST_YEAR = TODAY - timedelta(days=365)


@pytest.fixture(autouse=True)
def _fresh_dbs(tmp_path, monkeypatch):
    """Each test gets fresh ranger + permit databases."""
    monkeypatch.setenv("RANGERS_DB_PATH", str(tmp_path / "rangers.sqlite"))
    monkeypatch.setenv("PERMITS_DB_PATH", str(tmp_path / "permits.sqlite"))
    init_rangers_db()
    init_permits_db()


def _ar(label: str, confidence: float = 0.90) -> AudioResult:
    scores = {k: 0.0 for k in ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]}
    if label in scores:
        scores[label] = confidence
    return AudioResult(label=label, confidence=confidence, raw_scores=scores)


def _loc(lat: float, lon: float) -> TriangulationResult:
    return TriangulationResult(lat=lat, lon=lon, error_m=5.0)


# ---------------------------------------------------------------------------
# Ranger registration → alert routing
# ---------------------------------------------------------------------------


class TestRangerRegistrationFlow:
    """Simulates bot /start → district choice → ranger registered → alerts routed."""

    def test_ranger_registration_from_district(self) -> None:
        """Ranger registers for Varnavino district, gets correct zone."""
        district = DISTRICTS["varnavino"]
        ranger = add_ranger(
            name="Иванов И.И.",
            chat_id=123456,
            zone_lat_min=district.lat_min,
            zone_lat_max=district.lat_max,
            zone_lon_min=district.lon_min,
            zone_lon_max=district.lon_max,
        )
        assert ranger.zone_lat_min == district.lat_min
        assert ranger.zone_lat_max == district.lat_max

    def test_alert_routes_to_ranger_in_zone(self) -> None:
        """Threat inside ranger's zone → ranger receives alert."""
        district = DISTRICTS["varnavino"]
        add_ranger("Петров", 111, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)

        # Threat inside Varnavino
        lat, lon = 57.3, 45.0
        rangers = get_rangers_for_location(lat, lon)
        assert len(rangers) == 1
        assert rangers[0].chat_id == 111

    def test_alert_not_routed_outside_zone(self) -> None:
        """Threat outside ranger's zone → ranger does NOT receive alert."""
        district = DISTRICTS["varnavino"]
        add_ranger("Сидоров", 222, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)

        # Threat in Moscow (outside Varnavino)
        rangers = get_rangers_for_location(55.75, 37.61)
        assert len(rangers) == 0

    def test_multiple_rangers_same_zone(self) -> None:
        """Two rangers in same district → both get alerted."""
        district = DISTRICTS["varnavino"]
        add_ranger("Рейнджер 1", 333, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)
        add_ranger("Рейнджер 2", 444, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)

        rangers = get_rangers_for_location(57.3, 45.0)
        assert len(rangers) == 2


# ---------------------------------------------------------------------------
# Full pipeline: detection → decision → permit check
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: audio class + location → decider → permit check → alert or not."""

    def test_chainsaw_no_permit_alerts(self) -> None:
        """Chainsaw without permit → ALERT (illegal logging)."""
        loc = _loc(57.3, 45.0)
        decision = decide(_ar("chainsaw"), loc)
        assert decision.send_drone is True
        assert decision.priority == "high"

    def test_chainsaw_with_permit_suppressed(self) -> None:
        """Chainsaw with valid permit → NO alert (legal forestry)."""
        add_permit(57.0, 58.0, 44.5, 45.5, TODAY, NEXT_MONTH, "Делянка №5")
        loc = _loc(57.3, 45.0)
        decision = decide(_ar("chainsaw"), loc)
        assert decision.send_drone is False
        assert decision.send_lora is False
        assert "permit" in decision.reason.lower()

    def test_gunshot_with_permit_still_alerts(self) -> None:
        """Gunshot even WITH permit → ALERT (not covered by logging permit)."""
        add_permit(57.0, 58.0, 44.5, 45.5, TODAY, NEXT_MONTH, "Делянка №5")
        loc = _loc(57.3, 45.0)
        decision = decide(_ar("gunshot"), loc)
        assert decision.send_drone is True
        assert decision.priority == "high"

    def test_fire_with_permit_still_alerts(self) -> None:
        """Fire even WITH permit → ALERT (fires are never legal)."""
        add_permit(57.0, 58.0, 44.5, 45.5, TODAY, NEXT_MONTH)
        loc = _loc(57.3, 45.0)
        decision = decide(_ar("fire"), loc)
        assert decision.send_drone is True

    def test_expired_permit_chainsaw_alerts(self) -> None:
        """Expired permit + chainsaw → ALERT."""
        add_permit(57.0, 58.0, 44.5, 45.5, LAST_YEAR, YESTERDAY, "Истёк")
        loc = _loc(57.3, 45.0)
        decision = decide(_ar("chainsaw"), loc)
        assert decision.send_drone is True

    def test_permit_wrong_zone_chainsaw_alerts(self) -> None:
        """Permit in Moscow, chainsaw in Varnavino → ALERT."""
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH, "Московский")
        loc = _loc(57.3, 45.0)  # Varnavino
        decision = decide(_ar("chainsaw"), loc)
        assert decision.send_drone is True


# ---------------------------------------------------------------------------
# Combined: ranger + permit + threat
# ---------------------------------------------------------------------------


class TestRangerPermitIntegration:
    """Rangers registered, permits in place, threats detected — full flow."""

    def test_legal_logging_ranger_not_bothered(self) -> None:
        """Permit exists → chainsaw detected → ranger NOT notified."""
        district = DISTRICTS["varnavino"]
        add_ranger("Лесник Василий", 555, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)
        add_permit(57.0, 57.5, 44.8, 45.2, TODAY, NEXT_MONTH, "Санитарная рубка")

        loc = _loc(57.25, 45.0)
        decision = decide(_ar("chainsaw"), loc)
        # Decider says: no alert
        assert decision.send_drone is False
        assert decision.send_lora is False
        # Ranger exists for this zone but won't be bothered
        rangers = get_rangers_for_location(loc.lat, loc.lon)
        assert len(rangers) == 1  # ranger IS there
        # But the pipeline won't reach the notification step

    def test_illegal_logging_ranger_alerted(self) -> None:
        """No permit → chainsaw detected → ranger SHOULD be notified."""
        district = DISTRICTS["varnavino"]
        add_ranger("Лесник Василий", 666, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)
        # No permit added!

        loc = _loc(57.25, 45.0)
        decision = decide(_ar("chainsaw"), loc)
        assert decision.send_drone is True
        assert decision.send_lora is True

        # Ranger covers this zone — they will receive the alert
        rangers = get_rangers_for_location(loc.lat, loc.lon)
        assert len(rangers) == 1
        assert rangers[0].chat_id == 666

    def test_gunshot_in_permitted_zone_still_alerts_ranger(self) -> None:
        """Gunshot in zone with logging permit → ranger STILL notified."""
        district = DISTRICTS["varnavino"]
        add_ranger("Охотинспектор", 777, district.lat_min, district.lat_max,
                    district.lon_min, district.lon_max)
        add_permit(57.0, 57.5, 44.8, 45.2, TODAY, NEXT_MONTH, "Лесозаготовка")

        loc = _loc(57.25, 45.0)
        decision = decide(_ar("gunshot"), loc)
        assert decision.send_drone is True
        rangers = get_rangers_for_location(loc.lat, loc.lon)
        assert len(rangers) == 1
