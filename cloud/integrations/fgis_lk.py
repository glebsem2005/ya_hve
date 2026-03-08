"""FGIS-LK integration stub (ФГИС Лесного комплекса).

The real FGIS-LK does not expose a public API.  This module provides
data models and a mock client so that the codebase is ready for future
integration when the API becomes available.

All methods return realistic mock data and log warnings.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

logger = logging.getLogger(__name__)

_rng = random.Random(42)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ForestUnit:
    """Forest quarter (лесной квартал)."""

    quarter_number: int
    sub_district: str
    species_composition: str  # e.g. "6Е3Б1Ос"
    zone_type: str  # exploitation, oopt, water_protection, ...
    area_ha: float
    lat: float
    lon: float


@dataclass
class FellingPermit:
    """Felling declaration (лесная декларация)."""

    permit_id: str
    felling_type: str  # "sanitary", "commercial", "selective"
    zone_type: str
    volume_m3: float
    contractor: str
    valid_from: date
    valid_until: date
    quarter_number: int


@dataclass
class ViolationReport:
    """Violation report for submission to FGIS-LK."""

    incident_id: str
    audio_class: str
    lat: float
    lon: float
    confidence: float
    ranger_name: str
    description: str
    timestamp: str
    legal_articles: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Species composition templates
# ---------------------------------------------------------------------------

_SPECIES = [
    "6Е3Б1Ос",
    "7С2Е1Б",
    "5Б3Ос2Е",
    "8Е1Б1С",
    "4С3Е2Б1Ос",
    "9С1Б",
    "6Б2Е1Ос1Ол",
]

_FELLING_TYPES = ["sanitary", "commercial", "selective"]
_CONTRACTORS = [
    "ООО «ВарнаЛес»",
    "ИП Козлов А.С.",
    "ООО «НижЛесПром»",
    "ГБУ НО «Лесохрана»",
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FGISLKClient:
    """Mock FGIS-LK client.

    Every method returns plausible data and logs a warning that real
    API is not connected.
    """

    def get_forest_unit(self, lat: float, lon: float) -> ForestUnit:
        """Look up forest quarter by coordinates."""
        logger.warning("FGIS-LK stub: get_forest_unit(%.4f, %.4f)", lat, lon)
        return ForestUnit(
            quarter_number=_rng.randint(1, 420),
            sub_district="varnavinskoye",
            species_composition=_rng.choice(_SPECIES),
            zone_type="exploitation",
            area_ha=round(_rng.uniform(15.0, 120.0), 1),
            lat=lat,
            lon=lon,
        )

    def get_active_permits(self, lat: float, lon: float) -> list[FellingPermit]:
        """Get active felling permits covering the location."""
        logger.warning("FGIS-LK stub: get_active_permits(%.4f, %.4f)", lat, lon)
        # ~30% chance of having an active permit nearby
        if _rng.random() > 0.30:
            return []
        today = date.today()
        return [
            FellingPermit(
                permit_id=f"ЛД-{_rng.randint(1000, 9999)}/{today.year}",
                felling_type=_rng.choice(_FELLING_TYPES),
                zone_type="exploitation",
                volume_m3=round(_rng.uniform(50.0, 500.0), 1),
                contractor=_rng.choice(_CONTRACTORS),
                valid_from=today - timedelta(days=_rng.randint(10, 60)),
                valid_until=today + timedelta(days=_rng.randint(30, 180)),
                quarter_number=_rng.randint(1, 420),
            )
        ]

    def submit_violation(self, report: ViolationReport) -> dict:
        """Submit violation report to FGIS-LK (stub)."""
        logger.warning(
            "FGIS-LK stub: submit_violation(%s) -- not sent", report.incident_id
        )
        return {
            "status": "stub",
            "message": "FGIS-LK API not available; report stored locally",
            "report_id": f"STUB-{report.incident_id[:8]}",
        }

    def sync_permits(self, district_slug: str = "varnavino") -> list[FellingPermit]:
        """Sync permits from FGIS-LK for the district (stub)."""
        logger.warning("FGIS-LK stub: sync_permits(%s)", district_slug)
        today = date.today()
        n = _rng.randint(3, 8)
        return [
            FellingPermit(
                permit_id=f"ЛД-{_rng.randint(1000, 9999)}/{today.year}",
                felling_type=_rng.choice(_FELLING_TYPES),
                zone_type="exploitation",
                volume_m3=round(_rng.uniform(50.0, 500.0), 1),
                contractor=_rng.choice(_CONTRACTORS),
                valid_from=today - timedelta(days=_rng.randint(10, 90)),
                valid_until=today + timedelta(days=_rng.randint(30, 180)),
                quarter_number=_rng.randint(1, 420),
            )
            for _ in range(n)
        ]


# Module-level singleton
fgis_client = FGISLKClient()
