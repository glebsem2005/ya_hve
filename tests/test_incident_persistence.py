"""Tests for incident persistence: update_incident, get_all_incidents,
district auto-assignment, state machine, DataLens real data, concurrent accept.

These tests verify the unified DB layer (in-memory backend for tests).
"""

import time

import pytest

from cloud.db.incidents import (
    Incident,
    _incidents,
    _chat_to_incident,
    create_incident,
    get_incident,
    update_incident,
    get_all_incidents,
)
from cloud.analytics.datalens import get_datalens_incidents


@pytest.fixture(autouse=True)
def _fresh_incidents():
    """Clear incident state before each test."""
    _incidents.clear()
    _chat_to_incident.clear()
    yield
    _incidents.clear()
    _chat_to_incident.clear()


# ---------- TestUpdateIncidentPersistence ----------


class TestUpdateIncidentPersistence:
    def test_update_incident_exists(self):
        """update_incident is importable and callable."""
        assert callable(update_incident)

    def test_accept_persists_all_fields(self):
        """After accept, accepted_at and accepted_by_name are persisted."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(
            inc.id,
            status="accepted",
            accepted_at=123.0,
            accepted_by_name="Иванов",
        )
        reloaded = get_incident(inc.id)
        assert reloaded.accepted_at == 123.0
        assert reloaded.accepted_by_name == "Иванов"
        assert reloaded.status == "accepted"

    def test_update_multiple_fields(self):
        """Multiple fields updated in one call."""
        inc = create_incident("gunshot", 57.3, 45.0, 0.95, "alert")
        update_incident(
            inc.id,
            status="accepted",
            accepted_by_chat_id=42,
            accepted_by_name="Петров",
            accepted_at=100.0,
        )
        reloaded = get_incident(inc.id)
        assert reloaded.accepted_by_chat_id == 42
        assert reloaded.accepted_by_name == "Петров"
        assert reloaded.accepted_at == 100.0

    def test_update_nonexistent_incident(self):
        """Updating a nonexistent incident does not crash."""
        update_incident("nonexistent-id", status="accepted")  # no error


# ---------- TestGetAllIncidents ----------


class TestGetAllIncidents:
    def test_get_all_exists(self):
        """get_all_incidents is importable and callable."""
        assert callable(get_all_incidents)

    def test_returns_created_incidents(self):
        """All created incidents are returned."""
        create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        create_incident("gunshot", 57.4, 45.1, 0.8, "verify")
        all_inc = get_all_incidents()
        assert len(all_inc) == 2

    def test_empty_when_none_created(self):
        """Empty list when no incidents exist."""
        assert get_all_incidents() == []

    def test_order_by_created_at_desc(self):
        """Incidents are returned newest first."""
        inc1 = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        inc1.created_at = 100.0
        inc2 = create_incident("gunshot", 57.4, 45.1, 0.8, "verify")
        inc2.created_at = 200.0
        all_inc = get_all_incidents()
        assert all_inc[0].id == inc2.id
        assert all_inc[1].id == inc1.id


# ---------- TestIncidentDistrict ----------


class TestIncidentDistrict:
    def test_district_auto_assigned(self):
        """District is auto-assigned from coordinates."""
        inc = create_incident("chainsaw", 57.42, 44.70, 0.9, "alert")
        assert inc.district != ""
        assert "Мдальское" in inc.district

    def test_district_fallback_varnavino(self):
        """Coordinates outside sub-districts fall back to Варнавинское."""
        inc = create_incident("gunshot", 57.0, 44.5, 0.8, "alert")
        assert "Варнавинское" in inc.district

    def test_different_district_for_different_coords(self):
        """Different coordinates map to different districts."""
        inc1 = create_incident("chainsaw", 57.42, 44.70, 0.9, "alert")
        inc2 = create_incident("chainsaw", 57.12, 45.25, 0.9, "alert")
        assert inc1.district != inc2.district


# ---------- TestDatalensReturnsRealData ----------


class TestDatalensReturnsRealData:
    def test_datalens_returns_real_after_create(self):
        """DataLens returns real incidents from the DB."""
        create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        rows = get_datalens_incidents()
        assert any(
            r["audio_class"] == "chainsaw" and r["confidence"] == 0.9 for r in rows
        )

    def test_datalens_fallback_when_empty(self):
        """DataLens falls back to sample data when DB is empty."""
        rows = get_datalens_incidents()
        assert len(rows) == 200  # sample data

    def test_datalens_returns_only_real_when_nonempty(self):
        """When DB has incidents, only real ones are returned (no sample)."""
        create_incident("gunshot", 57.3, 45.0, 0.95, "alert")
        rows = get_datalens_incidents()
        assert len(rows) == 1
        assert rows[0]["audio_class"] == "gunshot"


# ---------- TestResponseTimePersistence ----------


class TestResponseTimePersistence:
    def test_response_time_persisted(self):
        """response_time_min is persisted via update_incident."""
        inc = create_incident("gunshot", 57.3, 45.0, 0.95, "alert")
        update_incident(inc.id, status="accepted")
        update_incident(
            inc.id,
            status="on_site",
            arrived_at=time.time(),
            response_time_min=12.5,
        )
        reloaded = get_incident(inc.id)
        assert reloaded.response_time_min == 12.5

    def test_arrived_at_persisted(self):
        """arrived_at is persisted via update_incident."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        now = time.time()
        update_incident(inc.id, status="accepted")
        update_incident(inc.id, status="on_site", arrived_at=now)
        reloaded = get_incident(inc.id)
        assert reloaded.arrived_at == now


# ---------- TestStateMachineValidation ----------


class TestStateMachineValidation:
    def test_cannot_revert_resolved_to_pending(self):
        """Cannot transition from resolved back to pending."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, status="accepted")
        update_incident(inc.id, status="on_site")
        update_incident(inc.id, status="resolved")
        # Attempt invalid transition
        update_incident(inc.id, status="pending")
        reloaded = get_incident(inc.id)
        assert reloaded.status == "resolved"

    def test_cannot_revert_false_alarm(self):
        """Cannot transition from false_alarm to any state."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, status="false_alarm")
        update_incident(inc.id, status="pending")
        reloaded = get_incident(inc.id)
        assert reloaded.status == "false_alarm"

    def test_valid_transition_pending_to_accepted(self):
        """Valid transition from pending to accepted."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, status="accepted")
        assert get_incident(inc.id).status == "accepted"

    def test_valid_transition_accepted_to_on_site(self):
        """Valid transition from accepted to on_site."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, status="accepted")
        update_incident(inc.id, status="on_site")
        assert get_incident(inc.id).status == "on_site"

    def test_valid_transition_on_site_to_resolved(self):
        """Valid transition from on_site to resolved."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, status="accepted")
        update_incident(inc.id, status="on_site")
        update_incident(inc.id, status="resolved")
        assert get_incident(inc.id).status == "resolved"

    def test_skip_pending_to_on_site_rejected(self):
        """Cannot skip accepted and go directly to on_site."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, status="on_site")
        assert get_incident(inc.id).status == "pending"

    def test_non_status_update_always_allowed(self):
        """Non-status fields can be updated regardless of state."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(inc.id, ranger_report_raw="Тест")
        assert get_incident(inc.id).ranger_report_raw == "Тест"


# ---------- TestConcurrentAccept ----------


class TestConcurrentAccept:
    def test_second_accept_rejected_by_state_machine(self):
        """Second accept is rejected because status is already 'accepted'."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(
            inc.id,
            status="accepted",
            accepted_by_name="Иванов",
            accepted_by_chat_id=100,
        )
        # Second ranger tries to accept — status already accepted,
        # transition accepted -> accepted is not in VALID_TRANSITIONS
        update_incident(
            inc.id,
            status="accepted",
            accepted_by_name="Петров",
            accepted_by_chat_id=200,
        )
        reloaded = get_incident(inc.id)
        # First ranger's data should remain
        assert reloaded.accepted_by_name == "Иванов"
        assert reloaded.accepted_by_chat_id == 100


# ---------- TestResolutionDetails ----------


class TestResolutionDetails:
    def test_resolution_details_persisted(self):
        """resolution_details field is persisted."""
        inc = create_incident("chainsaw", 57.3, 45.0, 0.9, "alert")
        update_incident(
            inc.id,
            resolution_details="Протокол составлен",
        )
        reloaded = get_incident(inc.id)
        assert reloaded.resolution_details == "Протокол составлен"
