"""Microphone network database — SQLite storage for mic nodes.

Each microphone has coordinates within Varnavino forestry district,
zone type (forest protection category), and operational status.

Microphones are placed on a diamond (rhombus) grid with ~350 m spacing
for maximum acoustic coverage of the district.
"""

import sqlite3
import math
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
    "shudskoye": {
        "name_ru": "Шудское",
        "lat_range": (57.67, 57.83),
        "lon_range": (44.32, 44.88),
    },
    "sivkoborskoye": {
        "name_ru": "Сивкоборское",
        "lat_range": (57.67, 57.83),
        "lon_range": (44.88, 45.44),
    },
    "lapshangskoye": {
        "name_ru": "Лапшангское",
        "lat_range": (57.52, 57.67),
        "lon_range": (44.32, 44.88),
    },
    "kameshnikovskoye": {
        "name_ru": "Камешниковское",
        "lat_range": (57.52, 57.67),
        "lon_range": (44.88, 45.44),
    },
    "khmelyevskoye": {
        "name_ru": "Хмелевское",
        "lat_range": (57.36, 57.52),
        "lon_range": (44.32, 44.88),
    },
    "varnavinskoye": {
        "name_ru": "Варнавинское",
        "lat_range": (57.36, 57.52),
        "lon_range": (44.88, 45.44),
    },
    "kalivetskoye": {
        "name_ru": "Каливецкое",
        "lat_range": (57.20, 57.36),
        "lon_range": (44.32, 44.88),
    },
    "kayskoye": {
        "name_ru": "Кайское",
        "lat_range": (57.20, 57.36),
        "lon_range": (44.88, 45.44),
    },
}

# Bounding box: Varnavino forestry district
LAT_MIN, LAT_MAX = 57.20, 57.83
LON_MIN, LON_MAX = 44.32, 45.44


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


def _build_diamond_grid(
    spacing_m: float = 350.0,
) -> list[tuple[float, float]]:
    """Build a diamond (rhombus) grid of (lat, lon) points inside Varnavino bbox.

    In a diamond grid odd rows are offset by half the column spacing,
    and the vertical row gap equals spacing × √3 / 2 ≈ 0.866 × spacing.
    This gives the densest uniform coverage for circular sensor footprints.

    Args:
        spacing_m: Distance between neighbouring microphones in metres
                   (300-400 m recommended for full acoustic coverage).

    Returns:
        List of (lat, lon) tuples covering the district.
    """
    centre_lat = (LAT_MIN + LAT_MAX) / 2  # ≈ 57.3°
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(centre_lat))

    row_step_deg = (spacing_m * math.sqrt(3) / 2) / m_per_deg_lat
    col_step_deg = spacing_m / m_per_deg_lon

    points: list[tuple[float, float]] = []
    row_idx = 0
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon_offset = (col_step_deg / 2) if (row_idx % 2 == 1) else 0.0
        lon = LON_MIN + lon_offset
        while lon <= LON_MAX:
            points.append((round(lat, 6), round(lon, 6)))
            lon += col_step_deg
        lat += row_step_deg
        row_idx += 1

    return points


# Default grid spacing in metres.
# 1 km side → each mic covers ≈ 1 km radius; ≈ 3 000 nodes for Varnavino.
GRID_SPACING_M = float(os.getenv("MIC_GRID_SPACING_M", "1000"))


def seed_microphones(
    spacing_m: float | None = None,
    seed: int = 42,
) -> list[Microphone]:
    """Populate the database with microphones on a diamond grid.

    Grid spacing is ~350 m by default (configurable via MIC_GRID_SPACING_M
    env-var or the *spacing_m* argument).  Zone types are distributed
    proportionally (~80 % exploitation, etc.).  ~15 % of nodes are randomly
    marked offline/broken.

    If the table is already populated the function returns the existing rows.
    """
    if spacing_m is None:
        spacing_m = GRID_SPACING_M

    rng = random.Random(seed)
    conn = _get_conn()

    # Don't re-seed if already populated
    count = conn.execute("SELECT COUNT(*) FROM microphones").fetchone()[0]
    if count > 0:
        rows = conn.execute("SELECT * FROM microphones").fetchall()
        conn.close()
        return [_row_to_mic(r) for r in rows]

    grid = _build_diamond_grid(spacing_m)

    mics: list[Microphone] = []
    for i, (lat, lon) in enumerate(grid, start=1):
        mic_uid = f"MIC-{i:04d}"
        zone_type = rng.choices(ZONE_TYPES, weights=ZONE_WEIGHTS, k=1)[0]
        sub_district = _assign_sub_district(lat, lon)

        # ~15 % offline / broken
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
                   (mic_uid, lat, lon, zone_type, sub_district, status,
                    battery_pct, district_slug, installed_at)
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


# ---------------------------------------------------------------------------
# Backend selection: YDB (cloud) or SQLite (local fallback)
# ---------------------------------------------------------------------------

if os.getenv("YDB_ENDPOINT"):
    from cloud.db.ydb_microphones import YDBMicrophoneRepository as _YDBRepo

    _repo = _YDBRepo()
    init_db = _repo.init_db
    seed_microphones = _repo.seed_microphones
    get_all = _repo.get_all
    get_online = _repo.get_online
    get_by_uid = _repo.get_by_uid
    set_status = _repo.set_status
    set_battery = _repo.set_battery
    clear_all = _repo.clear_all
else:
    # SQLite: auto-init on import
    init_db()

    def clear_all() -> int:
        """Delete all microphones from SQLite."""
        conn = _get_conn()
        cur = conn.execute("DELETE FROM microphones")
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count
