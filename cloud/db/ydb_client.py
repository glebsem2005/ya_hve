"""YDB connection client -- shared driver, session pool, and DDL management.

Only activated when YDB_ENDPOINT env var is set.
Gracefully handles missing ``ydb`` package with a clear error message.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

YDB_ENDPOINT: str = os.getenv("YDB_ENDPOINT", "")
YDB_DATABASE: str = os.getenv("YDB_DATABASE", "")
YDB_SA_KEY_FILE: str = os.getenv("YDB_SA_KEY_FILE", "")

_driver = None
_pool = None


# ---------------------------------------------------------------------------
# Connection management (singletons)
# ---------------------------------------------------------------------------


def get_driver():
    """Get or create YDB driver (singleton).

    Raises:
        ImportError: if ``ydb`` package is not installed.
        RuntimeError: if connection to YDB endpoint fails.
    """
    global _driver
    if _driver is not None:
        return _driver

    try:
        import ydb
    except ImportError:
        logger.error("ydb package not installed -- cannot use YDB backend")
        raise

    try:
        driver_config = ydb.DriverConfig(
            endpoint=YDB_ENDPOINT,
            database=YDB_DATABASE,
        )
        if YDB_SA_KEY_FILE:
            driver_config.credentials = ydb.iam.ServiceAccountCredentials.from_file(
                YDB_SA_KEY_FILE
            )
        _driver = ydb.Driver(driver_config)
        _driver.wait(timeout=5, fail_fast=True)
        logger.info("YDB driver connected to %s", YDB_ENDPOINT)
        return _driver
    except Exception as exc:
        logger.error("YDB connection failed: %s", exc)
        raise


def get_pool():
    """Get or create YDB session pool (singleton)."""
    global _pool
    if _pool is not None:
        return _pool

    try:
        import ydb
    except ImportError:
        logger.error("ydb package not installed -- cannot create session pool")
        raise

    try:
        driver = get_driver()
        _pool = ydb.SessionPool(driver, size=10)
        return _pool
    except Exception as exc:
        logger.error("YDB session pool creation failed: %s", exc)
        raise


def execute_query(session, sql: str, params: dict | None = None):
    """Execute a parameterized YDB query with automatic type binding.

    Uses ``session.prepare()`` which resolves DECLARE types from the SQL,
    so raw Python values (int, float, str, bool) are passed directly.
    """
    if params:
        prepared = session.prepare(sql)
        return session.transaction().execute(prepared, params, commit_tx=True)
    return session.transaction().execute(sql, commit_tx=True)


# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

DDL_RANGERS = """
CREATE TABLE rangers (
    id Uint64,
    name Utf8,
    badge_number Utf8,
    chat_id Int64,
    zone_lat_min Double,
    zone_lat_max Double,
    zone_lon_min Double,
    zone_lon_max Double,
    active Bool,
    PRIMARY KEY (id)
)
"""

DDL_PERMITS = """
CREATE TABLE permits (
    id Uint64,
    description Utf8,
    zone_lat_min Double,
    zone_lat_max Double,
    zone_lon_min Double,
    zone_lon_max Double,
    valid_from Utf8,
    valid_until Utf8,
    PRIMARY KEY (id)
)
"""

DDL_INCIDENTS = """
CREATE TABLE incidents (
    id Utf8,
    audio_class Utf8,
    lat Double,
    lon Double,
    confidence Double,
    gating_level Utf8,
    status Utf8,
    accepted_by_chat_id Int64,
    accepted_by_name Utf8,
    accepted_at Double,
    created_at Double,
    arrived_at Double,
    response_time_min Double,
    district Utf8,
    ranger_report_raw Utf8,
    ranger_report_legal Utf8,
    resolution_details Utf8,
    is_demo Bool,
    PRIMARY KEY (id)
)
"""

# ALTER TABLE statements for migrating existing tables
_ALTER_INCIDENTS = [
    "ALTER TABLE incidents ADD COLUMN created_at Double",
    "ALTER TABLE incidents ADD COLUMN arrived_at Double",
    "ALTER TABLE incidents ADD COLUMN response_time_min Double",
    "ALTER TABLE incidents ADD COLUMN district Utf8",
    "ALTER TABLE incidents ADD COLUMN ranger_report_raw Utf8",
    "ALTER TABLE incidents ADD COLUMN ranger_report_legal Utf8",
    "ALTER TABLE incidents ADD COLUMN resolution_details Utf8",
    "ALTER TABLE incidents ADD COLUMN is_demo Bool",
]

DDL_MICROPHONES = """
CREATE TABLE microphones (
    id Uint64,
    mic_uid Utf8,
    lat Double,
    lon Double,
    zone_type Utf8,
    sub_district Utf8,
    status Utf8,
    battery_pct Double,
    district_slug Utf8,
    installed_at Utf8,
    PRIMARY KEY (id)
)
"""


def ensure_tables() -> None:
    """Create all tables if they don't exist.

    Silently skips tables that are already present (catches SchemeError).
    Also runs ALTER TABLE migrations for existing incidents tables.
    Includes retry logic and throttling for YDB rate limits.
    """
    import time

    try:
        import ydb
    except ImportError:
        logger.error("ydb package not installed -- cannot ensure tables")
        raise

    try:
        pool = get_pool()
        for ddl in [DDL_RANGERS, DDL_PERMITS, DDL_INCIDENTS, DDL_MICROPHONES]:
            for attempt in range(3):
                try:
                    pool.retry_operation_sync(
                        lambda s, _ddl=ddl: s.execute_scheme(_ddl)
                    )
                    break
                except ydb.SchemeError:
                    break  # table already exists
                except Exception:
                    if attempt < 2:
                        time.sleep(2)
            time.sleep(0.5)  # throttle between DDL ops

        # Migrate existing incidents table — add new columns
        for alter in _ALTER_INCIDENTS:
            for attempt in range(3):
                try:
                    pool.retry_operation_sync(
                        lambda s, _alt=alter: s.execute_scheme(_alt)
                    )
                    break
                except ydb.SchemeError:
                    break  # column already exists
                except Exception:
                    if attempt < 2:
                        time.sleep(2)
            time.sleep(0.5)

        logger.info("YDB tables ensured")
    except Exception as exc:
        logger.error("Failed to ensure YDB tables: %s", exc)
