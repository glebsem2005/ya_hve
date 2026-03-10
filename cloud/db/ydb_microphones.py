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

GRID_SPACING_M = float(os.getenv("MIC_GRID_SPACING_M", "1000"))


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
            return self.get_all()

        rng = random.Random(seed)
        grid = _build_diamond_grid(spacing_m)
        mics: list[Microphone] = []

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

            def _ins(session, _i=i, _uid=mic_uid, _lat=lat, _lon=lon,
                     _zt=zone_type, _sd=sub_district, _st=status,
                     _bp=battery, _ia=installed_at):
                session.transaction().execute(
                    """
                    UPSERT INTO microphones (id, mic_uid, lat, lon,
                        zone_type, sub_district, status,
                        battery_pct, district_slug, installed_at)
                    VALUES ($id, $uid, $lat, $lon,
                        $zt, $sd, $st,
                        $bp, $ds, $ia)
                    """,
                    {
                        "$id": _i,
                        "$uid": _uid,
                        "$lat": _lat,
                        "$lon": _lon,
                        "$zt": _zt,
                        "$sd": _sd,
                        "$st": _st,
                        "$bp": _bp,
                        "$ds": "varnavino",
                        "$ia": _ia,
                    },
                    commit_tx=True,
                )

            pool.retry_operation_sync(_ins)

            mics.append(Microphone(
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
            ))

        return mics

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_all(self) -> list[Microphone]:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM microphones",
                commit_tx=True,
            )
            return [self._row_to_mic(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def get_online(self) -> list[Microphone]:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM microphones WHERE status = 'online'",
                commit_tx=True,
            )
            return [self._row_to_mic(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def get_by_uid(self, mic_uid: str) -> Microphone | None:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM microphones WHERE mic_uid = $uid",
                {"$uid": mic_uid},
                commit_tx=True,
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
            session.transaction().execute(
                "UPDATE microphones SET status = $st WHERE mic_uid = $uid",
                {"$st": status, "$uid": mic_uid},
                commit_tx=True,
            )

        pool.retry_operation_sync(_upd)
        return True

    def set_battery(self, mic_uid: str, battery_pct: float) -> bool:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()
        clamped = min(max(battery_pct, 0.0), 100.0)

        def _upd(session):
            session.transaction().execute(
                "UPDATE microphones SET battery_pct = $bp WHERE mic_uid = $uid",
                {"$bp": clamped, "$uid": mic_uid},
                commit_tx=True,
            )

        pool.retry_operation_sync(_upd)
        return True

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
