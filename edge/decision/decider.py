from __future__ import annotations

from dataclasses import dataclass

from edge.audio.classifier import AudioResult, AudioClass
from edge.audio.ndsi import NDSIResult
from edge.tdoa.triangulate import TriangulationResult
from cloud.db.permits import has_valid_permit

CONFIDENCE_ALERT = 0.70  # default threshold for most classes
CONFIDENCE_VERIFY = 0.40  # 0.40-threshold: only LoRa
# <0.40: log_only (13% accuracy) -> no action

# Per-class alert thresholds: critical threats (gunshot, fire) get a lower
# bar because the cost of missing them far exceeds the cost of a false flight.
ALERT_THRESHOLDS: dict[AudioClass, float] = {
    "gunshot": 0.55,
    "fire": 0.55,
    "chainsaw": 0.70,
    "axe": 0.70,
    "engine": 0.70,
    "background": 1.0,  # never triggers
    "unknown": 0.70,
}

PRIORITY_MAP: dict[AudioClass, str] = {
    "chainsaw": "high",
    "gunshot": "high",
    "fire": "high",
    "axe": "high",
    "engine": "medium",
    "background": "low",
    "unknown": "low",
}

SAFE_CLASSES: set[AudioClass] = {"background"}

# Classes that can be covered by a logging permit (лесной билет).
# If a valid permit exists at the detected location, these are legal.
PERMIT_CLASSES: set[AudioClass] = {"chainsaw", "axe", "engine"}


@dataclass
class Decision:
    send_drone: bool
    send_lora: bool
    priority: str
    reason: str


def _ndsi_suffix(ndsi: NDSIResult | None) -> str:
    """Build NDSI corroboration suffix for decision reason strings."""
    if ndsi is None or ndsi.ndsi >= -0.3:
        return ""
    return f" | NDSI={ndsi.ndsi:.2f} ({ndsi.interpretation})"


def decide(
    audio: AudioResult,
    location: TriangulationResult,
    ndsi: NDSIResult | None = None,
) -> Decision:
    """Make a gating decision based on classification, location, and optional NDSI.

    NDSI is informational only -- it corroborates anthropogenic detections
    when ndsi < -0.3 but does NOT change the gating thresholds.
    """
    if audio.label in SAFE_CLASSES:
        return Decision(
            send_drone=False,
            send_lora=False,
            priority="low",
            reason=f"Safe class detected: {audio.label}",
        )

    if audio.confidence < CONFIDENCE_VERIFY:
        return Decision(
            send_drone=False,
            send_lora=False,
            priority="low",
            reason=f"Log only: {audio.label} ({audio.confidence:.0%} < {CONFIDENCE_VERIFY:.0%})",
        )

    # Check logging permits for classes that could be legal forestry work
    if audio.label in PERMIT_CLASSES and has_valid_permit(location.lat, location.lon):
        return Decision(
            send_drone=False,
            send_lora=False,
            priority="low",
            reason=(
                f"Permitted activity: {audio.label} "
                f"({audio.confidence:.0%} confidence) "
                f"at {location.lat:.4f}°N {location.lon:.4f}°E — "
                f"valid logging permit found"
            ),
        )

    priority = PRIORITY_MAP.get(audio.label, "medium")
    suffix = _ndsi_suffix(ndsi)
    threshold = ALERT_THRESHOLDS.get(audio.label, CONFIDENCE_ALERT)

    if audio.confidence >= threshold:
        return Decision(
            send_drone=True,
            send_lora=True,
            priority=priority,
            reason=(
                f"Alert: {audio.label} "
                f"({audio.confidence:.0%} confidence) "
                f"at {location.lat:.4f}°N {location.lon:.4f}°E"
                f"{suffix}"
            ),
        )

    # Verify zone: 0.40 <= confidence < 0.70
    return Decision(
        send_drone=False,
        send_lora=True,
        priority=priority,
        reason=(
            f"Verify: {audio.label} "
            f"({audio.confidence:.0%} confidence) "
            f"at {location.lat:.4f}°N {location.lon:.4f}°E"
            f"{suffix}"
        ),
    )
