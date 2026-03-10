"""Forest logging permits (лесные билеты) — SQLite storage.

Each permit defines a geographic zone where logging is authorised
for a specific period.  When an acoustic event (chainsaw, axe, engine)
is detected, we check whether a valid permit covers that location.
If it does — the activity is legal and no alert is sent.
"""

import sqlite3
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_DEFAULT_DB_PATH = str(Path(__file__).parent / "permits.sqlite")


def _db_path() -> str:
    return os.getenv("PERMITS_DB_PATH", _DEFAULT_DB_PATH)


@dataclass
class Permit:
    id: int
    description: str
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float
    valid_from: date
    valid_until: date


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create permits table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS permits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL DEFAULT '',
            zone_lat_min REAL NOT NULL,
            zone_lat_max REAL NOT NULL,
            zone_lon_min REAL NOT NULL,
            zone_lon_max REAL NOT NULL,
            valid_from TEXT NOT NULL,
            valid_until TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_permit(
    zone_lat_min: float,
    zone_lat_max: float,
    zone_lon_min: float,
    zone_lon_max: float,
    valid_from: date,
    valid_until: date,
    description: str = "",
) -> Permit:
    """Register a new logging permit for a geographic zone."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO permits
           (description, zone_lat_min, zone_lat_max, zone_lon_min, zone_lon_max,
            valid_from, valid_until)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (description, zone_lat_min, zone_lat_max, zone_lon_min, zone_lon_max,
         valid_from.isoformat(), valid_until.isoformat()),
    )
    permit_id = cur.lastrowid
    conn.commit()
    conn.close()
    return Permit(
        id=permit_id,
        description=description,
        zone_lat_min=zone_lat_min,
        zone_lat_max=zone_lat_max,
        zone_lon_min=zone_lon_min,
        zone_lon_max=zone_lon_max,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def remove_permit(permit_id: int) -> bool:
    """Remove a permit by ID. Returns True if found and removed."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM permits WHERE id = ?", (permit_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_all_permits() -> list[Permit]:
    """Get all permits (active and expired)."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM permits").fetchall()
    conn.close()
    return [_row_to_permit(r) for r in rows]


def has_valid_permit(lat: float, lon: float, on_date: date | None = None) -> bool:
    """Check if there is a valid logging permit covering (lat, lon) on the given date.

    If on_date is None, uses today's date.
    """
    if on_date is None:
        on_date = date.today()
    d = on_date.isoformat()
    conn = _get_conn()
    row = conn.execute(
        """SELECT 1 FROM permits
           WHERE zone_lat_min <= ? AND zone_lat_max >= ?
             AND zone_lon_min <= ? AND zone_lon_max >= ?
             AND valid_from <= ? AND valid_until >= ?
           LIMIT 1""",
        (lat, lat, lon, lon, d, d),
    ).fetchone()
    conn.close()
    return row is not None


def get_permits_for_location(lat: float, lon: float, on_date: date | None = None) -> list[Permit]:
    """Find all valid permits covering (lat, lon) on the given date."""
    if on_date is None:
        on_date = date.today()
    d = on_date.isoformat()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM permits
           WHERE zone_lat_min <= ? AND zone_lat_max >= ?
             AND zone_lon_min <= ? AND zone_lon_max >= ?
             AND valid_from <= ? AND valid_until >= ?""",
        (lat, lat, lon, lon, d, d),
    ).fetchall()
    conn.close()
    return [_row_to_permit(r) for r in rows]


def _row_to_permit(row: sqlite3.Row) -> Permit:
    return Permit(
        id=row["id"],
        description=row["description"],
        zone_lat_min=row["zone_lat_min"],
        zone_lat_max=row["zone_lat_max"],
        zone_lon_min=row["zone_lon_min"],
        zone_lon_max=row["zone_lon_max"],
        valid_from=date.fromisoformat(row["valid_from"]),
        valid_until=date.fromisoformat(row["valid_until"]),
    )


# ---------------------------------------------------------------------------
# Backend selection: YDB (cloud) or SQLite (local fallback)
# ---------------------------------------------------------------------------

if os.getenv("YDB_ENDPOINT"):
    from cloud.db.ydb_permits import YDBPermitRepository as _YDBRepo

    _repo = _YDBRepo()
    init_db = _repo.init_db
    add_permit = _repo.add_permit
    remove_permit = _repo.remove_permit
    get_all_permits = _repo.get_all_permits
    has_valid_permit = _repo.has_valid_permit
    get_permits_for_location = _repo.get_permits_for_location
else:
    # SQLite: auto-init on import
    init_db()
