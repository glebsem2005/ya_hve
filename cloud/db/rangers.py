"""Ranger database — SQLite storage for forest rangers and their zones.

Each ranger has:
- telegram_chat_id: where to send alerts
- name: display name
- zone: geographic bounding box (lat_min, lat_max, lon_min, lon_max)
- current_lat/lon: last known position (for nearest-neighbor routing)
- active: whether to send alerts to this ranger

When an alert fires at (lat, lon), we find all rangers whose zone
contains that point and notify ONLY them.
"""

import math
import sqlite3
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DB_PATH = str(Path(__file__).parent / "rangers.sqlite")


def _db_path() -> str:
    return os.getenv("RANGERS_DB_PATH", _DEFAULT_DB_PATH)


@dataclass
class Ranger:
    id: int
    name: str
    badge_number: str
    chat_id: int
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float
    active: bool
    current_lat: float | None = None
    current_lon: float | None = None


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rangers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            badge_number TEXT NOT NULL DEFAULT '',
            chat_id INTEGER NOT NULL UNIQUE,
            zone_lat_min REAL NOT NULL DEFAULT 0,
            zone_lat_max REAL NOT NULL DEFAULT 90,
            zone_lon_min REAL NOT NULL DEFAULT 0,
            zone_lon_max REAL NOT NULL DEFAULT 180,
            active INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


def _migrate_db() -> None:
    """Add columns that may be missing in existing databases."""
    conn = _get_conn()
    for stmt in [
        "ALTER TABLE rangers ADD COLUMN badge_number TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE rangers ADD COLUMN current_lat REAL",
        "ALTER TABLE rangers ADD COLUMN current_lon REAL",
    ]:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.close()


def add_ranger(
    name: str,
    chat_id: int,
    zone_lat_min: float = 0.0,
    zone_lat_max: float = 90.0,
    zone_lon_min: float = 0.0,
    zone_lon_max: float = 180.0,
    badge_number: str = "",
) -> Ranger:
    """Add a new ranger. Returns the created Ranger."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO rangers (name, badge_number, chat_id, zone_lat_min, zone_lat_max, zone_lon_min, zone_lon_max)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            name,
            badge_number,
            chat_id,
            zone_lat_min,
            zone_lat_max,
            zone_lon_min,
            zone_lon_max,
        ),
    )
    ranger_id = cur.lastrowid
    conn.commit()
    conn.close()
    return Ranger(
        id=ranger_id,
        name=name,
        badge_number=badge_number,
        chat_id=chat_id,
        zone_lat_min=zone_lat_min,
        zone_lat_max=zone_lat_max,
        zone_lon_min=zone_lon_min,
        zone_lon_max=zone_lon_max,
        active=True,
        current_lat=None,
        current_lon=None,
    )


def remove_ranger(chat_id: int) -> bool:
    """Remove a ranger by chat_id. Returns True if found and removed."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM rangers WHERE chat_id = ?", (chat_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def set_active(chat_id: int, active: bool) -> bool:
    """Enable/disable alerts for a ranger."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE rangers SET active = ? WHERE chat_id = ?",
        (int(active), chat_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def update_zone(
    chat_id: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> bool:
    """Update a ranger's monitoring zone."""
    conn = _get_conn()
    cur = conn.execute(
        """UPDATE rangers
           SET zone_lat_min = ?, zone_lat_max = ?, zone_lon_min = ?, zone_lon_max = ?
           WHERE chat_id = ?""",
        (lat_min, lat_max, lon_min, lon_max, chat_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def update_position(chat_id: int, lat: float, lon: float) -> bool:
    """Save ranger's current GPS position."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE rangers SET current_lat = ?, current_lon = ? WHERE chat_id = ?",
        (lat, lon, chat_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_nearest_rangers(lat: float, lon: float, limit: int = 3) -> list[Ranger]:
    """Return nearest active rangers sorted by haversine distance.

    Only includes rangers with a known position (current_lat/lon not NULL).
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM rangers WHERE active = 1 AND current_lat IS NOT NULL"
    ).fetchall()
    conn.close()
    rangers = [_row_to_ranger(r) for r in rows]
    rangers.sort(key=lambda r: _haversine(lat, lon, r.current_lat, r.current_lon))
    return rangers[:limit]


def get_all_rangers() -> list[Ranger]:
    """Get all rangers."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM rangers").fetchall()
    conn.close()
    return [_row_to_ranger(r) for r in rows]


def get_rangers_for_location(lat: float, lon: float) -> list[Ranger]:
    """Find all active rangers whose zone contains the given coordinates."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM rangers
           WHERE active = 1
             AND zone_lat_min <= ? AND zone_lat_max >= ?
             AND zone_lon_min <= ? AND zone_lon_max >= ?""",
        (lat, lat, lon, lon),
    ).fetchall()
    conn.close()
    return [_row_to_ranger(r) for r in rows]


def get_ranger_by_chat_id(chat_id: int) -> Ranger | None:
    """Find a ranger by their Telegram chat ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM rangers WHERE chat_id = ?", (chat_id,)).fetchone()
    conn.close()
    return _row_to_ranger(row) if row else None


def _row_to_ranger(row: sqlite3.Row) -> Ranger:
    return Ranger(
        id=row["id"],
        name=row["name"],
        badge_number=row["badge_number"],
        chat_id=row["chat_id"],
        zone_lat_min=row["zone_lat_min"],
        zone_lat_max=row["zone_lat_max"],
        zone_lon_min=row["zone_lon_min"],
        zone_lon_max=row["zone_lon_max"],
        active=bool(row["active"]),
        current_lat=row["current_lat"],
        current_lon=row["current_lon"],
    )


# ---------------------------------------------------------------------------
# Backend selection: YDB (cloud) or SQLite (local fallback)
# ---------------------------------------------------------------------------

if os.getenv("YDB_ENDPOINT"):
    from cloud.db.ydb_rangers import YDBRangerRepository as _YDBRepo

    _repo = _YDBRepo()
    init_db = _repo.init_db
    add_ranger = _repo.add_ranger
    remove_ranger = _repo.remove_ranger
    set_active = _repo.set_active
    update_zone = _repo.update_zone
    get_all_rangers = _repo.get_all_rangers
    get_rangers_for_location = _repo.get_rangers_for_location
    get_ranger_by_chat_id = _repo.get_ranger_by_chat_id
else:
    # SQLite: auto-init on import
    init_db()
    _migrate_db()
