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


def get_datalens_stats() -> dict:
    """Return aggregated statistics for DataLens dashboard."""
    rows = get_datalens_incidents()

    # By class
    by_class: dict[str, int] = {}
    for r in rows:
        cls = r["audio_class"]
        by_class[cls] = by_class.get(cls, 0) + 1

    # By status
    by_status: dict[str, int] = {}
    for r in rows:
        st = r["status"]
        by_status[st] = by_status.get(st, 0) + 1

    # By district
    by_district: dict[str, int] = {}
    for r in rows:
        d = r["district"]
        by_district[d] = by_district.get(d, 0) + 1

    # Average response time (resolved only)
    resolved = [r for r in rows if r["response_time_min"] is not None]
    avg_response = (
        round(sum(r["response_time_min"] for r in resolved) / len(resolved), 1)
        if resolved
        else 0
    )

    # Incidents per day (last 30 days)
    total = len(rows)
    daily_avg = round(total / 30, 1)

    return {
        "total_incidents": total,
        "by_class": by_class,
        "by_status": by_status,
        "by_district": by_district,
        "avg_response_time_min": avg_response,
        "daily_average": daily_avg,
        "detection_rate": round(by_status.get("resolved", 0) / max(total, 1) * 100, 1),
        "false_alarm_rate": round(
            by_status.get("false_alarm", 0) / max(total, 1) * 100, 1
        ),
    }
