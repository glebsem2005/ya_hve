"""YDB implementation of PermitRepository.

Mirrors the SQLite permit API (``cloud.db.permits``) but persists data
in Yandex Database.  Activated only when ``YDB_ENDPOINT`` env var is set.
"""

from __future__ import annotations

import logging
from datetime import date

from cloud.db.base import PermitRepository
from cloud.db.permits import Permit

logger = logging.getLogger(__name__)

# Auto-increment counter for permit IDs (YDB has no AUTOINCREMENT).
_next_permit_id: int = 1


class YDBPermitRepository(PermitRepository):
    """YDB-backed permit storage."""

    def __init__(self) -> None:
        from cloud.db.ydb_client import ensure_tables

        try:
            ensure_tables()
            self._sync_next_id()
        except Exception as exc:
            logger.warning("YDB init failed, operations may fail: %s", exc)

    def _sync_next_id(self) -> None:
        """Read max existing ID from YDB so we don't collide."""
        global _next_permit_id
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT MAX(id) AS max_id FROM permits",
                commit_tx=True,
            )
            rows = result[0].rows
            if rows and rows[0].max_id is not None:
                return int(rows[0].max_id) + 1
            return 1

        try:
            _next_permit_id = pool.retry_operation_sync(_q)
        except Exception:
            _next_permit_id = 1

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """No-op -- table creation handled by ``ensure_tables``."""

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_permit(
        self,
        zone_lat_min: float,
        zone_lat_max: float,
        zone_lon_min: float,
        zone_lon_max: float,
        valid_from: date,
        valid_until: date,
        description: str = "",
    ) -> Permit:
        global _next_permit_id
        from cloud.db.ydb_client import get_pool

        pool = get_pool()
        permit_id = _next_permit_id
        _next_permit_id += 1

        def _add(session):
            session.transaction().execute(
                """
                UPSERT INTO permits (id, description,
                    zone_lat_min, zone_lat_max, zone_lon_min, zone_lon_max,
                    valid_from, valid_until)
                VALUES ($id, $desc,
                    $lat_min, $lat_max, $lon_min, $lon_max,
                    $vfrom, $vuntil)
                """,
                {
                    "$id": permit_id,
                    "$desc": description,
                    "$lat_min": zone_lat_min,
                    "$lat_max": zone_lat_max,
                    "$lon_min": zone_lon_min,
                    "$lon_max": zone_lon_max,
                    "$vfrom": valid_from.isoformat(),
                    "$vuntil": valid_until.isoformat(),
                },
                commit_tx=True,
            )

        pool.retry_operation_sync(_add)
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

    def remove_permit(self, permit_id: int) -> bool:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _rm(session):
            session.transaction().execute(
                "DELETE FROM permits WHERE id = $pid",
                {"$pid": permit_id},
                commit_tx=True,
            )

        pool.retry_operation_sync(_rm)
        return True

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_all_permits(self) -> list[Permit]:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM permits",
                commit_tx=True,
            )
            return [self._row_to_permit(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def has_valid_permit(
        self, lat: float, lon: float, on_date: date | None = None
    ) -> bool:
        if on_date is None:
            on_date = date.today()
        d = on_date.isoformat()

        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                """SELECT id FROM permits
                   WHERE zone_lat_min <= $lat AND zone_lat_max >= $lat
                     AND zone_lon_min <= $lon AND zone_lon_max >= $lon
                     AND valid_from <= $d AND valid_until >= $d
                   LIMIT 1""",
                {"$lat": lat, "$lon": lon, "$d": d},
                commit_tx=True,
            )
            return len(result[0].rows) > 0

        return pool.retry_operation_sync(_q)

    def get_permits_for_location(
        self, lat: float, lon: float, on_date: date | None = None
    ) -> list[Permit]:
        if on_date is None:
            on_date = date.today()
        d = on_date.isoformat()

        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                """SELECT * FROM permits
                   WHERE zone_lat_min <= $lat AND zone_lat_max >= $lat
                     AND zone_lon_min <= $lon AND zone_lon_max >= $lon
                     AND valid_from <= $d AND valid_until >= $d""",
                {"$lat": lat, "$lon": lon, "$d": d},
                commit_tx=True,
            )
            return [self._row_to_permit(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_permit(row) -> Permit:
        return Permit(
            id=row.id,
            description=row.description,
            zone_lat_min=row.zone_lat_min,
            zone_lat_max=row.zone_lat_max,
            zone_lon_min=row.zone_lon_min,
            zone_lon_max=row.zone_lon_max,
            valid_from=date.fromisoformat(row.valid_from),
            valid_until=date.fromisoformat(row.valid_until),
        )
