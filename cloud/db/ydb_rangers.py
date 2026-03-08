"""YDB implementation of RangerRepository.

Mirrors the SQLite ranger API (``cloud.db.rangers``) but persists data
in Yandex Database.  Activated only when ``YDB_ENDPOINT`` env var is set.
"""

from __future__ import annotations

import logging

from cloud.db.base import RangerRepository
from cloud.db.rangers import Ranger

logger = logging.getLogger(__name__)


class YDBRangerRepository(RangerRepository):
    """YDB-backed ranger storage."""

    def __init__(self) -> None:
        from cloud.db.ydb_client import ensure_tables

        try:
            ensure_tables()
        except Exception as exc:
            logger.warning("YDB init failed, operations may fail: %s", exc)

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """No-op -- table creation handled by ``ensure_tables``."""

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_ranger(
        self,
        name: str,
        chat_id: int,
        zone_lat_min: float = 0.0,
        zone_lat_max: float = 90.0,
        zone_lon_min: float = 0.0,
        zone_lon_max: float = 180.0,
        badge_number: str = "",
    ) -> Ranger:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _add(session):
            session.transaction().execute(
                """
                UPSERT INTO rangers (id, name, badge_number, chat_id,
                    zone_lat_min, zone_lat_max, zone_lon_min, zone_lon_max, active)
                VALUES ($id, $name, $badge, $chat_id,
                    $lat_min, $lat_max, $lon_min, $lon_max, true)
                """,
                {
                    "$id": chat_id,
                    "$name": name,
                    "$badge": badge_number,
                    "$chat_id": chat_id,
                    "$lat_min": zone_lat_min,
                    "$lat_max": zone_lat_max,
                    "$lon_min": zone_lon_min,
                    "$lon_max": zone_lon_max,
                },
                commit_tx=True,
            )

        pool.retry_operation_sync(_add)
        return Ranger(
            id=chat_id,
            name=name,
            badge_number=badge_number,
            chat_id=chat_id,
            zone_lat_min=zone_lat_min,
            zone_lat_max=zone_lat_max,
            zone_lon_min=zone_lon_min,
            zone_lon_max=zone_lon_max,
            active=True,
        )

    def remove_ranger(self, chat_id: int) -> bool:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _rm(session):
            session.transaction().execute(
                "DELETE FROM rangers WHERE chat_id = $cid",
                {"$cid": chat_id},
                commit_tx=True,
            )

        pool.retry_operation_sync(_rm)
        return True

    def set_active(self, chat_id: int, active: bool) -> bool:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _upd(session):
            session.transaction().execute(
                "UPDATE rangers SET active = $active WHERE chat_id = $cid",
                {"$active": active, "$cid": chat_id},
                commit_tx=True,
            )

        pool.retry_operation_sync(_upd)
        return True

    def update_zone(
        self,
        chat_id: int,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> bool:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _upd(session):
            session.transaction().execute(
                """UPDATE rangers
                   SET zone_lat_min = $lamin, zone_lat_max = $lamax,
                       zone_lon_min = $lomin, zone_lon_max = $lomax
                   WHERE chat_id = $cid""",
                {
                    "$lamin": lat_min,
                    "$lamax": lat_max,
                    "$lomin": lon_min,
                    "$lomax": lon_max,
                    "$cid": chat_id,
                },
                commit_tx=True,
            )

        pool.retry_operation_sync(_upd)
        return True

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_all_rangers(self) -> list[Ranger]:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM rangers",
                commit_tx=True,
            )
            return [self._row_to_ranger(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def get_rangers_for_location(self, lat: float, lon: float) -> list[Ranger]:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                """SELECT * FROM rangers
                   WHERE active = true
                     AND zone_lat_min <= $lat AND zone_lat_max >= $lat
                     AND zone_lon_min <= $lon AND zone_lon_max >= $lon""",
                {"$lat": lat, "$lon": lon},
                commit_tx=True,
            )
            return [self._row_to_ranger(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def get_ranger_by_chat_id(self, chat_id: int) -> Ranger | None:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM rangers WHERE chat_id = $cid",
                {"$cid": chat_id},
                commit_tx=True,
            )
            rows = result[0].rows
            return self._row_to_ranger(rows[0]) if rows else None

        return pool.retry_operation_sync(_q)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_ranger(row) -> Ranger:
        return Ranger(
            id=row.id,
            name=row.name,
            badge_number=row.badge_number,
            chat_id=row.chat_id,
            zone_lat_min=row.zone_lat_min,
            zone_lat_max=row.zone_lat_max,
            zone_lon_min=row.zone_lon_min,
            zone_lon_max=row.zone_lon_max,
            active=row.active,
        )
