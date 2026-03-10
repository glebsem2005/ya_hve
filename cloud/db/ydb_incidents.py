"""YDB implementation of IncidentRepository.

Makes incidents **persistent** across restarts (unlike the in-memory default).
Chat-to-incident mapping remains in-memory for simplicity during demo --
it is transient by nature (cleared when a ranger finishes workflow).
"""

from __future__ import annotations

import logging
import time
import uuid

from cloud.db.base import IncidentRepository
from cloud.db.incidents import Incident

logger = logging.getLogger(__name__)

# Chat-to-incident mapping: transient, in-memory (same as SQLite version).
_chat_to_incident: dict[int, str] = {}


class YDBIncidentRepository(IncidentRepository):
    """YDB-backed incident storage with in-memory chat mapping."""

    def __init__(self) -> None:
        from cloud.db.ydb_client import ensure_tables

        try:
            ensure_tables()
        except Exception as exc:
            logger.warning("YDB init failed, operations may fail: %s", exc)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def create_incident(
        self,
        audio_class: str,
        lat: float,
        lon: float,
        confidence: float,
        gating_level: str,
        is_demo: bool = False,
    ) -> Incident:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()
        incident_id = str(uuid.uuid4())

        def _ins(session):
            session.transaction().execute(
                """
                UPSERT INTO incidents (id, audio_class, lat, lon,
                    confidence, gating_level, status,
                    accepted_by_chat_id, accepted_by_name, accepted_at)
                VALUES ($id, $cls, $lat, $lon,
                    $conf, $gate, $status,
                    $chat_id, $name, $at)
                """,
                {
                    "$id": incident_id,
                    "$cls": audio_class,
                    "$lat": lat,
                    "$lon": lon,
                    "$conf": confidence,
                    "$gate": gating_level,
                    "$status": "pending",
                    "$chat_id": 0,
                    "$name": "",
                    "$at": 0.0,
                },
                commit_tx=True,
            )

        pool.retry_operation_sync(_ins)

        return Incident(
            id=incident_id,
            audio_class=audio_class,
            lat=lat,
            lon=lon,
            confidence=confidence,
            gating_level=gating_level,
            status="pending",
            is_demo=is_demo,
        )

    def update_status(self, incident_id: str, status: str) -> None:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _upd(session):
            session.transaction().execute(
                "UPDATE incidents SET status = $status WHERE id = $id",
                {"$status": status, "$id": incident_id},
                commit_tx=True,
            )

        pool.retry_operation_sync(_upd)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_incident(self, incident_id: str) -> Incident | None:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            result = session.transaction().execute(
                "SELECT * FROM incidents WHERE id = $id",
                {"$id": incident_id},
                commit_tx=True,
            )
            rows = result[0].rows
            return self._row_to_incident(rows[0]) if rows else None

        return pool.retry_operation_sync(_q)

    def get_active_incident_for_chat(self, chat_id: int) -> Incident | None:
        """Look up active incident via in-memory chat mapping, then fetch from YDB."""
        incident_id = _chat_to_incident.get(chat_id)
        if incident_id is None:
            return None
        return self.get_incident(incident_id)

    # ------------------------------------------------------------------
    # chat mapping (in-memory, transient)
    # ------------------------------------------------------------------

    def assign_chat_to_incident(self, chat_id: int, incident_id: str) -> None:
        _chat_to_incident[chat_id] = incident_id

    def clear_chat_incident(self, chat_id: int) -> None:
        _chat_to_incident.pop(chat_id, None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_incident(row) -> Incident:
        """Convert a YDB result row to an Incident dataclass.

        Note: fields that only exist in-memory (drone_photo_b64, ranger_photo_b64,
        protocol_pdf, etc.) are not stored in YDB and default to None.
        """
        return Incident(
            id=row.id,
            audio_class=row.audio_class,
            lat=row.lat,
            lon=row.lon,
            confidence=row.confidence,
            gating_level=row.gating_level,
            status=row.status,
            accepted_by_chat_id=row.accepted_by_chat_id
            if row.accepted_by_chat_id
            else None,
            accepted_by_name=row.accepted_by_name if row.accepted_by_name else None,
            accepted_at=row.accepted_at if row.accepted_at else None,
        )
