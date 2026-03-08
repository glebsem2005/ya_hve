from dataclasses import dataclass
from edge.audio.classifier import AudioResult, AudioClass
from edge.tdoa.triangulate import TriangulationResult

CONFIDENCE_ALERT = 0.70   # >0.70: 95% accuracy -> drone + LoRa
CONFIDENCE_VERIFY = 0.40  # 0.40-0.70: 49% accuracy -> only LoRa
# <0.40: log_only (13% accuracy) -> no action

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


@dataclass
class Decision:
    send_drone: bool
    send_lora: bool
    priority: str
    reason: str


def decide(audio: AudioResult, location: TriangulationResult) -> Decision:
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

    priority = PRIORITY_MAP.get(audio.label, "medium")

    if audio.confidence >= CONFIDENCE_ALERT:
        return Decision(
            send_drone=True,
            send_lora=True,
            priority=priority,
            reason=(
                f"Alert: {audio.label} "
                f"({audio.confidence:.0%} confidence) "
                f"at {location.lat:.4f}°N {location.lon:.4f}°E"
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
        ),
    )
