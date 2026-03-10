"""Database layer — auto-selects YDB (cloud) or SQLite (local) backend.

When ``YDB_ENDPOINT`` env var is set, all modules (rangers, permits,
incidents, microphones) transparently delegate to Yandex Database.
Otherwise, local SQLite files are used as fallback.
"""

import logging
import os

logger = logging.getLogger(__name__)

BACKEND = "ydb" if os.getenv("YDB_ENDPOINT") else "sqlite"
logger.info("Database backend: %s", BACKEND)
