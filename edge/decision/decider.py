from dataclasses import dataclass
from edge.audio.classifier import AudioResult, AudioClass
from edge.tdoa.triangulate import TriangulationResult

CONFIDENCE_THRESHOLD = 0.70

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

    if audio.confidence < CONFIDENCE_THRESHOLD:
        return Decision(
            send_drone=False,
            send_lora=False,
            priority="low",
            reason=f"Confidence too low: {audio.confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%}",
        )

    priority = PRIORITY_MAP.get(audio.label, "medium")

    return Decision(
        send_drone=True,
        send_lora=True,
        priority=priority,
        reason=(
            f"Anomaly detected: {audio.label} "
            f"({audio.confidence:.0%} confidence) "
            f"at {location.lat:.4f}°N {location.lon:.4f}°E"
        ),
    )
