"""DataLens dashboard data endpoints.

Provides JSON-formatted data suitable for Yandex DataLens API Connector.
Includes aggregated statistics by class, district, status, and response time.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from cloud.analytics.sample_incidents import generate_incidents

_SEED = 42


def get_datalens_incidents() -> list[dict]:
    """Return enriched incidents for DataLens.

    Adds response_time_min, ranger_name, and resolution_details
    on top of the base incident data.
    """
    rng = random.Random(_SEED)
    rows = generate_incidents(n=50)

    ranger_names = [
        "Козлов А.С.",
        "Петров И.В.",
        "Сидорова Е.М.",
        "Васильев Д.А.",
        "Кузнецов П.Н.",
    ]

    for row in rows:
        if row["status"] == "resolved":
            row["response_time_min"] = rng.randint(8, 90)
            row["ranger_name"] = rng.choice(ranger_names)
            row["resolution_details"] = "Протокол составлен, материалы переданы"
        elif row["status"] == "false_alarm":
            row["response_time_min"] = rng.randint(5, 45)
            row["ranger_name"] = rng.choice(ranger_names)
            row["resolution_details"] = "Ложное срабатывание, закрыто"
        else:
            row["response_time_min"] = None
            row["ranger_name"] = None
            row["resolution_details"] = "Ожидает реакции"

    return rows


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
