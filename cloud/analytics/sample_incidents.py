"""Generate sample incident data for DataLens dashboard.

Creates a CSV with 50 realistic forest monitoring incidents
in the Varnavino forestry district (Nizhny Novgorod Oblast).

Usage:
    python -m cloud.analytics.sample_incidents
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

CLASSES = ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]
CLASS_WEIGHTS = [0.40, 0.20, 0.15, 0.10, 0.10, 0.05]

STATUSES = ["resolved", "false_alarm", "pending"]
STATUS_WEIGHTS = [0.60, 0.25, 0.15]

GATING_LEVELS = ["alert", "verify", "log"]

DISTRICTS = [
    "Мдальское",
    "Семёнборское",
    "Поплывинское",
    "Каменниковское",
    "Варнавинское",
    "Колесниковское",
    "Камешное",
    "Кайское",
]

RANGER_NAMES = [
    "Козлов А.С.",
    "Петров И.В.",
    "Сидорова Е.М.",
    "Васильев Д.А.",
    "Кузнецов П.Н.",
]

# Varnavino forestry zone bounding box
LAT_MIN, LAT_MAX = 57.0, 57.5
LON_MIN, LON_MAX = 44.5, 45.5

OUTPUT_PATH = Path(__file__).parent / "sample_incidents.csv"
NUM_INCIDENTS = 50


def _random_timestamp(days_back: int = 30) -> str:
    base = datetime(2026, 3, 8, 12, 0, 0)
    offset = timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(5, 22),
        minutes=random.randint(0, 59),
    )
    return (base - offset).strftime("%Y-%m-%d %H:%M:%S")


def _confidence_for_class(cls: str) -> float:
    if cls == "background":
        return round(random.uniform(0.30, 0.55), 3)
    return round(random.uniform(0.60, 0.98), 3)


def _gating_for_confidence(conf: float) -> str:
    if conf >= 0.85:
        return "alert"
    if conf >= 0.65:
        return "verify"
    return "log"


def generate_incidents(n: int = NUM_INCIDENTS) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        cls = random.choices(CLASSES, weights=CLASS_WEIGHTS, k=1)[0]
        conf = _confidence_for_class(cls)
        status = random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        if cls == "background":
            status = "false_alarm"

        if status == "resolved":
            response_time = random.randint(8, 90)
            ranger_name = random.choice(RANGER_NAMES)
            resolution = "Протокол составлен, материалы переданы"
        elif status == "false_alarm":
            response_time = random.randint(5, 45)
            ranger_name = random.choice(RANGER_NAMES)
            resolution = "Ложное срабатывание, закрыто"
        else:
            response_time = None
            ranger_name = None
            resolution = "Ожидает реакции"

        rows.append(
            {
                "id": i,
                "timestamp": _random_timestamp(),
                "lat": round(random.uniform(LAT_MIN, LAT_MAX), 6),
                "lon": round(random.uniform(LON_MIN, LON_MAX), 6),
                "audio_class": cls,
                "confidence": conf,
                "gating_level": _gating_for_confidence(conf),
                "status": status,
                "district": random.choice(DISTRICTS),
                "response_time_min": response_time,
                "ranger_name": ranger_name,
                "resolution_details": resolution,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path = OUTPUT_PATH) -> None:
    fieldnames = [
        "id",
        "timestamp",
        "lat",
        "lon",
        "audio_class",
        "confidence",
        "gating_level",
        "status",
        "district",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_incidents_csv_text() -> str:
    """Return CSV content as string (for API export)."""
    rows = generate_incidents()
    import io

    buf = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    random.seed(42)
    rows = generate_incidents()
    write_csv(rows)
    print(f"Generated {len(rows)} incidents -> {OUTPUT_PATH}")
