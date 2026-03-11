"""Incident store with state machine for ranger workflow.

States: pending -> accepted -> on_site -> resolved | false_alarm

Backend selection: YDB (when YDB_ENDPOINT is set) or in-memory (default/tests).
"""

import time
import uuid
from dataclasses import dataclass, field

from cloud.notify.districts import get_district_name

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"accepted", "false_alarm"},
    "accepted": {"on_site", "false_alarm"},
    "on_site": {"resolved", "false_alarm"},
    "resolved": set(),
    "false_alarm": set(),
}

# Fields that update_incident is allowed to write
_UPDATABLE_FIELDS = frozenset(
    {
        "status",
        "accepted_by_chat_id",
        "accepted_by_name",
        "accepted_at",
        "arrived_at",
        "response_time_min",
        "drone_photo_b64",
        "drone_comment",
        "ranger_photo_b64",
        "ranger_report_raw",
        "ranger_report_legal",
        "protocol_pdf",
        "resolution_details",
        "district",
        "is_demo",
    }
)


@dataclass
class Incident:
    id: str
    audio_class: str
    lat: float
    lon: float
    confidence: float
    gating_level: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    district: str = ""
    drone_photo_b64: str | None = None
    drone_comment: str | None = None
    accepted_by_chat_id: int | None = None
    accepted_by_name: str | None = None
    accepted_at: float | None = None
    arrived_at: float | None = None
    response_time_min: float | None = None
    alert_message_ids: dict[int, int] = field(default_factory=dict)
    ranger_photo_b64: str | None = None
    ranger_report_raw: str | None = None
    ranger_report_legal: str | None = None
    protocol_pdf: bytes | None = None
    resolution_details: str = ""
    is_demo: bool = False


# ---------------------------------------------------------------------------
# In-memory backend (used by default / tests)
# ---------------------------------------------------------------------------

# incident_id -> Incident
_incidents: dict[str, Incident] = {}

# chat_id -> incident_id (active incident for contextual messages)
_chat_to_incident: dict[int, str] = {}


def create_incident(
    audio_class: str,
    lat: float,
    lon: float,
    confidence: float,
    gating_level: str,
    is_demo: bool = False,
) -> Incident:
    incident = Incident(
        id=str(uuid.uuid4()),
        audio_class=audio_class,
        lat=lat,
        lon=lon,
        confidence=confidence,
        gating_level=gating_level,
        district=get_district_name(lat, lon),
        is_demo=is_demo,
    )
    _incidents[incident.id] = incident
    return incident


def get_incident(incident_id: str) -> Incident | None:
    return _incidents.get(incident_id)


def get_active_incident_for_chat(chat_id: int) -> Incident | None:
    incident_id = _chat_to_incident.get(chat_id)
    if incident_id:
        return _incidents.get(incident_id)
    return None


def assign_chat_to_incident(chat_id: int, incident_id: str) -> None:
    _chat_to_incident[chat_id] = incident_id


def clear_chat_incident(chat_id: int) -> None:
    _chat_to_incident.pop(chat_id, None)


def update_status(incident_id: str, status: str) -> None:
    incident = _incidents.get(incident_id)
    if incident:
        incident.status = status


def update_incident(incident_id: str, **fields) -> None:
    """Update incident fields with state machine validation.

    Only fields in _UPDATABLE_FIELDS are accepted.
    Status transitions are validated against VALID_TRANSITIONS.
    """
    incident = _incidents.get(incident_id)
    if not incident:
        return

    new_status = fields.get("status")
    if new_status:
        if new_status == incident.status:
            # Same status = no-op (protects against concurrent accept)
            return
        allowed = VALID_TRANSITIONS.get(incident.status, set())
        if new_status not in allowed:
            return  # reject invalid transition

    for key, value in fields.items():
        if key in _UPDATABLE_FIELDS:
            setattr(incident, key, value)


def get_all_incidents() -> list[Incident]:
    """Return all incidents ordered by created_at descending."""
    return sorted(_incidents.values(), key=lambda i: i.created_at, reverse=True)


def clear_all_incidents() -> None:
    """Clear all incidents and chat mappings (for tests)."""
    _incidents.clear()
    _chat_to_incident.clear()


def get_stale_incidents(
    pending_max_age: float = 1800, accepted_max_age: float = 3600
) -> list[Incident]:
    """Find stale incidents: pending > pending_max_age, accepted > accepted_max_age."""
    now = time.time()
    stale = []
    for incident in _incidents.values():
        if incident.status == "pending":
            if now - incident.created_at > pending_max_age:
                stale.append(incident)
        elif incident.status == "accepted" and incident.accepted_at:
            if now - incident.accepted_at > accepted_max_age:
                stale.append(incident)
    return stale


def get_recent_nearby_incident(
    lat: float, lon: float, radius_m: float = 500, max_age_s: float = 300
) -> Incident | None:
    """Find a recent pending/accepted incident near the given coordinates."""
    from cloud.db.rangers import _haversine

    now = time.time()
    for incident in _incidents.values():
        if incident.status not in ("pending", "accepted"):
            continue
        if now - incident.created_at > max_age_s:
            continue
        if _haversine(lat, lon, incident.lat, incident.lon) <= radius_m:
            return incident
    return None


# ---------------------------------------------------------------------------
# Backend selection: YDB (cloud) or in-memory (default)
# ---------------------------------------------------------------------------

import os as _os

if _os.getenv("YDB_ENDPOINT"):
    from cloud.db.ydb_incidents import YDBIncidentRepository as _YDBRepo

    _repo = _YDBRepo()
    create_incident = _repo.create_incident
    get_incident = _repo.get_incident
    get_active_incident_for_chat = _repo.get_active_incident_for_chat
    assign_chat_to_incident = _repo.assign_chat_to_incident
    clear_chat_incident = _repo.clear_chat_incident
    update_status = _repo.update_status
    update_incident = _repo.update_incident
    get_all_incidents = _repo.get_all_incidents
    get_stale_incidents = getattr(_repo, "get_stale_incidents", get_stale_incidents)
    get_recent_nearby_incident = getattr(
        _repo, "get_recent_nearby_incident", get_recent_nearby_incident
    )
