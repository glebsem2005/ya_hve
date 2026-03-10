"""Abstract repository interfaces for database backends.

Supports SQLite (default) and YDB (when configured via YDB_ENDPOINT env var).
All existing code continues to use SQLite -- YDB is opt-in.

Pattern: Repository + Factory.  Each entity (ranger, permit, incident,
microphone) has an abstract base class here.  Concrete implementations
live in the corresponding sqlite / ydb modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloud.db.incidents import Incident
    from cloud.db.microphones import Microphone
    from cloud.db.permits import Permit
    from cloud.db.rangers import Ranger


# ---------------------------------------------------------------------------
# Rangers
# ---------------------------------------------------------------------------


class RangerRepository(ABC):
    """Abstract ranger storage."""

    @abstractmethod
    def init_db(self) -> None: ...

    @abstractmethod
    def add_ranger(
        self,
        name: str,
        chat_id: int,
        zone_lat_min: float,
        zone_lat_max: float,
        zone_lon_min: float,
        zone_lon_max: float,
        badge_number: str = "",
    ) -> Ranger: ...

    @abstractmethod
    def remove_ranger(self, chat_id: int) -> bool: ...

    @abstractmethod
    def set_active(self, chat_id: int, active: bool) -> bool: ...

    @abstractmethod
    def update_zone(
        self,
        chat_id: int,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> bool: ...

    @abstractmethod
    def get_all_rangers(self) -> list[Ranger]: ...

    @abstractmethod
    def get_rangers_for_location(self, lat: float, lon: float) -> list[Ranger]: ...

    @abstractmethod
    def get_ranger_by_chat_id(self, chat_id: int) -> Ranger | None: ...


# ---------------------------------------------------------------------------
# Permits
# ---------------------------------------------------------------------------


class PermitRepository(ABC):
    """Abstract permit storage."""

    @abstractmethod
    def init_db(self) -> None: ...

    @abstractmethod
    def add_permit(
        self,
        zone_lat_min: float,
        zone_lat_max: float,
        zone_lon_min: float,
        zone_lon_max: float,
        valid_from: date,
        valid_until: date,
        description: str = "",
    ) -> Permit: ...

    @abstractmethod
    def remove_permit(self, permit_id: int) -> bool: ...

    @abstractmethod
    def get_all_permits(self) -> list[Permit]: ...

    @abstractmethod
    def has_valid_permit(
        self, lat: float, lon: float, on_date: date | None = None
    ) -> bool: ...

    @abstractmethod
    def get_permits_for_location(
        self, lat: float, lon: float, on_date: date | None = None
    ) -> list[Permit]: ...


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------


class IncidentRepository(ABC):
    """Abstract incident storage."""

    @abstractmethod
    def create_incident(
        self,
        audio_class: str,
        lat: float,
        lon: float,
        confidence: float,
        gating_level: str,
        is_demo: bool = False,
    ) -> Incident: ...

    @abstractmethod
    def get_incident(self, incident_id: str) -> Incident | None: ...

    @abstractmethod
    def get_active_incident_for_chat(self, chat_id: int) -> Incident | None: ...

    @abstractmethod
    def assign_chat_to_incident(self, chat_id: int, incident_id: str) -> None: ...

    @abstractmethod
    def clear_chat_incident(self, chat_id: int) -> None: ...

    @abstractmethod
    def update_status(self, incident_id: str, status: str) -> None: ...


# ---------------------------------------------------------------------------
# Microphones
# ---------------------------------------------------------------------------


class MicrophoneRepository(ABC):
    """Abstract microphone storage."""

    @abstractmethod
    def init_db(self) -> None: ...

    @abstractmethod
    def seed_microphones(self, n: int = 20, seed: int = 42) -> list[Microphone]: ...

    @abstractmethod
    def get_all(self) -> list[Microphone]: ...

    @abstractmethod
    def get_online(self) -> list[Microphone]: ...

    @abstractmethod
    def get_by_uid(self, mic_uid: str) -> Microphone | None: ...

    @abstractmethod
    def set_status(self, mic_uid: str, status: str) -> bool: ...

    @abstractmethod
    def set_battery(self, mic_uid: str, battery_pct: float) -> bool: ...
