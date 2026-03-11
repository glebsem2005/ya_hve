"""DataLens dashboard data endpoints.

Provides JSON-formatted data suitable for Yandex DataLens API Connector.
Uses real incidents from the database; falls back to sample data when empty.
"""

from __future__ import annotations

from datetime import datetime

from cloud.db.incidents import get_all_incidents


def _incident_to_dict(inc) -> dict:
    """Convert an Incident dataclass to a DataLens-friendly dict."""
    return {
        "id": inc.id,
        "timestamp": datetime.fromtimestamp(inc.created_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        if inc.created_at
        else "",
        "lat": inc.lat,
        "lon": inc.lon,
        "audio_class": inc.audio_class,
        "confidence": inc.confidence,
        "gating_level": inc.gating_level,
        "status": inc.status,
        "district": inc.district or "",
        "response_time_min": inc.response_time_min,
        "ranger_name": inc.accepted_by_name or "",
        "resolution_details": inc.resolution_details or "",
    }


def get_datalens_incidents() -> list[dict]:
    """Return incidents for DataLens.

    Uses real incidents from the database.
    Falls back to sample data only when the DB is empty (first launch / demo).
    """
    incidents = get_all_incidents()
    if incidents:
        return [_incident_to_dict(inc) for inc in incidents]

    # Fallback: seed data for empty DB (demo / first launch)
    from cloud.analytics.sample_incidents import generate_incidents

    return generate_incidents(n=200, seed=42)
