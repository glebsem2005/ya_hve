"""Microphone network database — SQLite storage for mic nodes.

Each microphone has coordinates within Varnavino forestry district,
zone type (forest protection category), and operational status.
"""

import sqlite3
import os
import random
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DB_PATH = str(Path(__file__).parent / "microphones.sqlite")


def _db_path() -> str:
    return os.getenv("MICS_DB_PATH", _DEFAULT_DB_PATH)


@dataclass
class Microphone:
    id: int
    mic_uid: str
    lat: float
    lon: float
    zone_type: str
    sub_district: str
    status: str
    battery_pct: float
    district_slug: str
    installed_at: str


ZONE_TYPES = [
    "exploitation",
    "oopt",
    "water_protection",
    "protective_strip",
    "green_zone",
    "water_restricted",
    "spawning_protection",
    "anti_erosion",
]

# Distribution weights matching Varnavino map (~80% exploitation, etc.)
ZONE_WEIGHTS = [0.80, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02]

SUB_DISTRICTS = {
    "mdalskoe": {
        "name_ru": "Мдальское",
        "lat_range": (57.40, 57.55),
        "lon_range": (44.60, 44.80),
    },
    "semyonborskoe": {
        "name_ru": "Семёнборское",
        "lat_range": (57.35, 57.50),
        "lon_range": (44.80, 45.00),
    },
    "poplyvinskoye": {
        "name_ru": "Поплывинское",
        "lat_range": (57.30, 57.45),
        "lon_range": (45.00, 45.20),
    },
    "kamennikoskoye": {
        "name_ru": "Каменниковское",
        "lat_range": (57.20, 57.35),
        "lon_range": (44.60, 44.80),
    },
    "varnavinskoye": {
        "name_ru": "Варнавинское",
        "lat_range": (57.15, 57.30),
        "lon_range": (44.80, 45.00),
    },
    "kolesnikovskoye": {
        "name_ru": "Колесниковское",
        "lat_range": (57.10, 57.25),
        "lon_range": (45.00, 45.20),
    },
    "kameshnoye": {
        "name_ru": "Камешное",
        "lat_range": (57.05, 57.20),
        "lon_range": (45.10, 45.30),
    },
    "kayskoye": {
        "name_ru": "Кайское",
        "lat_range": (57.05, 57.20),
        "lon_range": (45.20, 45.40),
    },
}

# Bounding box: Varnavino forestry district
LAT_MIN, LAT_MAX = 57.05, 57.55
LON_MIN, LON_MAX = 44.60, 45.40


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS microphones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mic_uid TEXT NOT NULL UNIQUE,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            zone_type TEXT NOT NULL DEFAULT 'exploitation',
            sub_district TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'online',
            battery_pct REAL NOT NULL DEFAULT 100.0,
            district_slug TEXT NOT NULL DEFAULT 'varnavino',
            installed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _assign_sub_district(lat: float, lon: float) -> str:
    """Assign sub-district based on coordinates matching approximate quadrants."""
    for slug, info in SUB_DISTRICTS.items():
        lat_lo, lat_hi = info["lat_range"]
        lon_lo, lon_hi = info["lon_range"]
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return slug
    return "varnavinskoye"  # default


def seed_microphones(n: int = 20, seed: int = 42) -> list[Microphone]:
    """Generate N microphones with random coordinates inside Varnavino bbox.

    Zone type distributed proportionally (~80% exploitation, etc.).
    ~15% randomly offline/broken.
    """
    rng = random.Random(seed)

    conn = _get_conn()
    # Don't re-seed if already populated
    count = conn.execute("SELECT COUNT(*) FROM microphones").fetchone()[0]
    if count >= n:
        rows = conn.execute("SELECT * FROM microphones").fetchall()
        conn.close()
        return [_row_to_mic(r) for r in rows]

    mics = []
    for i in range(1, n + 1):
        mic_uid = f"MIC-{i:03d}"
        lat = round(rng.uniform(LAT_MIN, LAT_MAX), 6)
        lon = round(rng.uniform(LON_MIN, LON_MAX), 6)
        zone_type = rng.choices(ZONE_TYPES, weights=ZONE_WEIGHTS, k=1)[0]
        sub_district = _assign_sub_district(lat, lon)

        # ~15% offline/broken
        status_roll = rng.random()
        if status_roll < 0.10:
            status = "offline"
        elif status_roll < 0.15:
            status = "broken"
        else:
            status = "online"

        battery = round(rng.uniform(20.0, 100.0), 1)
        installed_at = f"2026-{rng.randint(1, 3):02d}-{rng.randint(1, 28):02d}"

        try:
            conn.execute(
                """INSERT OR IGNORE INTO microphones
                   (mic_uid, lat, lon, zone_type, sub_district, status, battery_pct, district_slug, installed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mic_uid,
                    lat,
                    lon,
                    zone_type,
                    sub_district,
                    status,
                    battery,
                    "varnavino",
                    installed_at,
                ),
            )
        except sqlite3.IntegrityError:
            pass

        mics.append(
            Microphone(
                id=i,
                mic_uid=mic_uid,
                lat=lat,
                lon=lon,
                zone_type=zone_type,
                sub_district=sub_district,
                status=status,
                battery_pct=battery,
                district_slug="varnavino",
                installed_at=installed_at,
            )
        )

    conn.commit()
    conn.close()
    return mics


def get_all() -> list[Microphone]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM microphones").fetchall()
    conn.close()
    return [_row_to_mic(r) for r in rows]


def get_online() -> list[Microphone]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM microphones WHERE status = 'online'").fetchall()
    conn.close()
    return [_row_to_mic(r) for r in rows]


def get_by_uid(mic_uid: str) -> Microphone | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM microphones WHERE mic_uid = ?", (mic_uid,)
    ).fetchone()
    conn.close()
    return _row_to_mic(row) if row else None


def set_status(mic_uid: str, status: str) -> bool:
    if status not in ("online", "offline", "broken"):
        return False
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE microphones SET status = ? WHERE mic_uid = ?", (status, mic_uid)
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def set_battery(mic_uid: str, battery_pct: float) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE microphones SET battery_pct = ? WHERE mic_uid = ?",
        (min(max(battery_pct, 0.0), 100.0), mic_uid),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def _row_to_mic(row: sqlite3.Row) -> Microphone:
    return Microphone(
        id=row["id"],
        mic_uid=row["mic_uid"],
        lat=row["lat"],
        lon=row["lon"],
        zone_type=row["zone_type"],
        sub_district=row["sub_district"],
        status=row["status"],
        battery_pct=row["battery_pct"],
        district_slug=row["district_slug"],
        installed_at=row["installed_at"],
    )


# Auto-init on import
init_db()
