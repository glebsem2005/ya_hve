"""YDB implementation of MicrophoneRepository.

Mirrors the SQLite microphone API (``cloud.db.microphones``) but persists data
in Yandex Database.  Activated only when ``YDB_ENDPOINT`` env var is set.
"""

from __future__ import annotations

import logging
import math
import os
import random

from cloud.db.base import MicrophoneRepository
from cloud.db.ydb_client import execute_query
from cloud.db.microphones import (
    Microphone,
    ZONE_TYPES,
    ZONE_WEIGHTS,
    SUB_DISTRICTS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    _assign_sub_district,
    _build_diamond_grid,
)

logger = logging.getLogger(__name__)

GRID_SPACING_M = float(os.getenv("MIC_GRID_SPACING_M", "1500"))


class YDBMicrophoneRepository(MicrophoneRepository):
    """YDB-backed microphone storage."""

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
    # seed
    # ------------------------------------------------------------------

    def seed_microphones(
        self,
        spacing_m: float | None = None,
        seed: int = 42,
    ) -> list[Microphone]:
        """Populate YDB with microphones on a diamond grid.

        If the table is already populated, returns the existing rows.
        Uses bulk_upsert for fast batch insertion.
        """
        if spacing_m is None:
            spacing_m = GRID_SPACING_M

        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        # Check if already populated
        def _count(session):
            result = session.transaction().execute(
                "SELECT COUNT(*) AS cnt FROM microphones",
                commit_tx=True,
            )
            return result[0].rows[0].cnt

        count = pool.retry_operation_sync(_count)
        if count > 0:
            logger.info("Microphones already seeded (%d rows), skipping", count)
            return []

        import ydb
        from cloud.db.ydb_client import get_driver, YDB_DATABASE

        rng = random.Random(seed)
        grid = _build_diamond_grid(spacing_m)
        mics: list[Microphone] = []

        column_types = (
            ydb.BulkUpsertColumns()
            .add_column("id", ydb.PrimitiveType.Uint64)
            .add_column("mic_uid", ydb.PrimitiveType.Utf8)
            .add_column("lat", ydb.PrimitiveType.Double)
            .add_column("lon", ydb.PrimitiveType.Double)
            .add_column("zone_type", ydb.PrimitiveType.Utf8)
            .add_column("sub_district", ydb.PrimitiveType.Utf8)
            .add_column("status", ydb.PrimitiveType.Utf8)
            .add_column("battery_pct", ydb.PrimitiveType.Double)
            .add_column("district_slug", ydb.PrimitiveType.Utf8)
            .add_column("installed_at", ydb.PrimitiveType.Utf8)
        )

        rows: list[dict] = []
        for i, (lat, lon) in enumerate(grid, start=1):
            mic_uid = f"MIC-{i:04d}"
            zone_type = rng.choices(ZONE_TYPES, weights=ZONE_WEIGHTS, k=1)[0]
            sub_district = _assign_sub_district(lat, lon)

            status_roll = rng.random()
            if status_roll < 0.10:
                status = "offline"
            elif status_roll < 0.15:
                status = "broken"
            else:
                status = "online"

            battery = round(rng.uniform(20.0, 100.0), 1)
            installed_at = f"2026-{rng.randint(1, 3):02d}-{rng.randint(1, 28):02d}"

            row = {
                "id": i,
                "mic_uid": mic_uid,
                "lat": lat,
                "lon": lon,
                "zone_type": zone_type,
                "sub_district": sub_district,
                "status": status,
                "battery_pct": battery,
                "district_slug": "varnavino",
                "installed_at": installed_at,
            }
            rows.append(row)

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

        # Bulk upsert in batches with retry and delay for YDB rate limits
        import time

        driver = get_driver()
        table_path = f"{YDB_DATABASE}/microphones"
        batch_size = 500
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for attempt in range(5):
                try:
                    driver.table_client.bulk_upsert(table_path, batch, column_types)
                    logger.info(
                        "Bulk upserted mics %d-%d",
                        start + 1,
                        start + len(batch),
                    )
                    break
                except Exception as exc:
                    if attempt < 4:
                        wait = 2 ** (attempt + 1)
                        logger.warning(
                            "Batch %d failed (%s), retry in %ds",
                            start // batch_size,
                            exc,
                            wait,
                        )
                        time.sleep(wait)
                    else:
                        raise
            time.sleep(1.0)  # throttle between batches

        logger.info("Seeded %d microphones via bulk_upsert", len(mics))
        return mics

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_all(self, limit: int = 10000) -> list[Microphone]:
        from cloud.db.ydb_client import get_driver, YDB_DATABASE

        driver = get_driver()
        table_path = f"{YDB_DATABASE}/microphones"

        # Use scan query to avoid truncation on large result sets
        mics: list[Microphone] = []
        it = driver.table_client.scan_query(f"SELECT * FROM microphones LIMIT {limit}")
        while True:
            try:
                result = next(it)
                for row in result.result_set.rows:
                    mics.append(self._row_to_mic(row))
            except StopIteration:
                break

        return mics

    def get_online(self, limit: int = 100) -> list[Microphone]:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            from cloud.db.ydb_client import execute_query

            result = execute_query(
                session,
                f"DECLARE $lim AS Uint64; SELECT * FROM microphones WHERE status = 'online' LIMIT $lim",
                {"$lim": limit},
            )
            return [self._row_to_mic(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def get_by_uid(self, mic_uid: str) -> Microphone | None:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = execute_query(
                session,
                "DECLARE $uid AS Utf8; SELECT * FROM microphones WHERE mic_uid = $uid",
                {"$uid": mic_uid},
            )
            rows = result[0].rows
            return self._row_to_mic(rows[0]) if rows else None

        return pool.retry_operation_sync(_q)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def set_status(self, mic_uid: str, status: str) -> bool:
        if status not in ("online", "offline", "broken"):
            return False

        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _upd(session):
            execute_query(
                session,
                "DECLARE $st AS Utf8; DECLARE $uid AS Utf8; UPDATE microphones SET status = $st WHERE mic_uid = $uid",
                {"$st": status, "$uid": mic_uid},
            )

        pool.retry_operation_sync(_upd)
        return True

    def set_battery(self, mic_uid: str, battery_pct: float) -> bool:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()
        clamped = min(max(battery_pct, 0.0), 100.0)

        def _upd(session):
            execute_query(
                session,
                "DECLARE $bp AS Double; DECLARE $uid AS Utf8; UPDATE microphones SET battery_pct = $bp WHERE mic_uid = $uid",
                {"$bp": clamped, "$uid": mic_uid},
            )

        pool.retry_operation_sync(_upd)
        return True

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def clear_all(self) -> int:
        """Delete all rows from the microphones table."""
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _count(session):
            result = session.transaction().execute(
                "SELECT COUNT(*) AS cnt FROM microphones",
                commit_tx=True,
            )
            return result[0].rows[0].cnt

        count = pool.retry_operation_sync(_count)

        def _delete(session):
            session.transaction().execute(
                "DELETE FROM microphones",
                commit_tx=True,
            )

        pool.retry_operation_sync(_delete)
        return count

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_mic(row) -> Microphone:
        return Microphone(
            id=row.id,
            mic_uid=row.mic_uid,
            lat=row.lat,
            lon=row.lon,
            zone_type=row.zone_type,
            sub_district=row.sub_district,
            status=row.status,
            battery_pct=row.battery_pct,
            district_slug=row.district_slug,
            installed_at=row.installed_at,
        )
