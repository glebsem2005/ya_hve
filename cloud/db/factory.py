"""Database backend factory.

If YDB_ENDPOINT env var is set, returns YDB repository implementations.
Otherwise, returns None -- callers fall back to existing SQLite module-level
functions (full backward compatibility, zero changes to existing code).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloud.db.base import (
        IncidentRepository,
        MicrophoneRepository,
        PermitRepository,
        RangerRepository,
    )

logger = logging.getLogger(__name__)

_backend: str | None = None


def _detect_backend() -> str:
    """Detect database backend from environment."""
    if os.getenv("YDB_ENDPOINT"):
        return "ydb"
    return "sqlite"


def get_backend() -> str:
    """Return active backend name (cached after first call)."""
    global _backend
    if _backend is None:
        _backend = _detect_backend()
        logger.info("Database backend: %s", _backend)
    return _backend


def get_ranger_repository() -> RangerRepository | None:
    """Get ranger repository for the active backend.

    Returns None for SQLite -- caller should use module-level functions
    from ``cloud.db.rangers`` directly (backward compatible).
    """
    if get_backend() == "ydb":
        from cloud.db.ydb_rangers import YDBRangerRepository

        return YDBRangerRepository()
    return None


def get_permit_repository() -> PermitRepository | None:
    """Get permit repository for the active backend.

    Returns None for SQLite -- caller should use module-level functions
    from ``cloud.db.permits`` directly.
    """
    if get_backend() == "ydb":
        from cloud.db.ydb_permits import YDBPermitRepository

        return YDBPermitRepository()
    return None


def get_incident_repository() -> IncidentRepository | None:
    """Get incident repository for the active backend.

    Returns None for SQLite -- caller should use module-level functions
    from ``cloud.db.incidents`` directly.
    """
    if get_backend() == "ydb":
        from cloud.db.ydb_incidents import YDBIncidentRepository

        return YDBIncidentRepository()
    return None


def get_microphone_repository() -> MicrophoneRepository | None:
    """Get microphone repository for the active backend.

    Returns None for SQLite -- caller should use module-level functions
    from ``cloud.db.microphones`` directly.
    """
    if get_backend() == "ydb":
        # YDB microphone repo not implemented yet
        logger.warning("YDB MicrophoneRepository not implemented, using SQLite")
        return None
    return None
