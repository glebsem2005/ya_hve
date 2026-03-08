"""In-memory incident store with state machine for ranger workflow.

States: pending -> accepted -> on_site -> resolved | false_alarm

For demo purposes, uses in-memory dict (not persistent across restarts).
"""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Incident:
    id: str
    audio_class: str
    lat: float
    lon: float
    confidence: float
    gating_level: str
    status: str = "pending"
    drone_photo_b64: str | None = None
    drone_comment: str | None = None
    accepted_by_chat_id: int | None = None
    accepted_by_name: str | None = None
    accepted_at: float | None = None
    alert_message_ids: dict[int, int] = field(default_factory=dict)
    ranger_photo_b64: str | None = None
    ranger_report_raw: str | None = None
    ranger_report_legal: str | None = None
    protocol_pdf: bytes | None = None
    is_demo: bool = False


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
