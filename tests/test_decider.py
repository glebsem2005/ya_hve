"""Tests for edge.decision.decider — the decision engine.

17 tests covering priority mapping, confidence thresholds, safe-class
bypass, and output dataclass structure.
"""

from __future__ import annotations

import pytest

from edge.audio.classifier import AudioResult
from edge.decision.decider import (
    CONFIDENCE_THRESHOLD,
    Decision,
    PRIORITY_MAP,
    SAFE_CLASSES,
    decide,
)
from edge.tdoa.triangulate import TriangulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ar(label: str, confidence: float) -> AudioResult:
    """Shorthand to build an AudioResult for tests."""
    scores = {
        "chainsaw": 0.0,
        "gunshot": 0.0,
        "engine": 0.0,
        "axe": 0.0,
        "fire": 0.0,
        "background": 0.0,
    }
    if label in scores:
        scores[label] = confidence
    return AudioResult(label=label, confidence=confidence, raw_scores=scores)


def _loc(lat: float = 55.7510, lon: float = 37.6130) -> TriangulationResult:
    return TriangulationResult(lat=lat, lon=lon, error_m=5.0)


# ---------------------------------------------------------------------------
# Safe-class bypass
# ---------------------------------------------------------------------------


class TestSafeClass:
    def test_safe_class_background_no_drone(self) -> None:
        decision = decide(_ar("background", 0.99), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False
        assert decision.priority == "low"

    def test_safe_classes_only_background(self) -> None:
        assert SAFE_CLASSES == {"background"}

    def test_no_silence_in_safe_classes(self) -> None:
        assert "silence" not in SAFE_CLASSES


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_low_confidence_no_drone(self) -> None:
        decision = decide(_ar("chainsaw", 0.50), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False
        assert decision.priority == "low"

    def test_confidence_at_threshold(self) -> None:
        decision = decide(_ar("chainsaw", CONFIDENCE_THRESHOLD), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True

    def test_confidence_below_threshold(self) -> None:
        decision = decide(_ar("chainsaw", 0.69), _loc())
        assert decision.send_drone is False
        assert decision.send_lora is False


# ---------------------------------------------------------------------------
# Priority mapping — high
# ---------------------------------------------------------------------------


class TestHighPriority:
    @pytest.mark.parametrize("label", ["chainsaw", "gunshot", "fire", "axe"])
    def test_high_priority(self, label: str) -> None:
        decision = decide(_ar(label, 0.90), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "high"

    def test_high_priority_chainsaw(self) -> None:
        decision = decide(_ar("chainsaw", 0.90), _loc())
        assert decision.priority == "high"

    def test_high_priority_gunshot(self) -> None:
        decision = decide(_ar("gunshot", 0.90), _loc())
        assert decision.priority == "high"

    def test_high_priority_fire(self) -> None:
        decision = decide(_ar("fire", 0.90), _loc())
        assert decision.priority == "high"

    def test_high_priority_axe(self) -> None:
        decision = decide(_ar("axe", 0.90), _loc())
        assert decision.priority == "high"


# ---------------------------------------------------------------------------
# Medium / low priorities
# ---------------------------------------------------------------------------


class TestOtherPriorities:
    def test_medium_priority_engine(self) -> None:
        decision = decide(_ar("engine", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.priority == "medium"

    def test_unknown_low_confidence(self) -> None:
        decision = decide(_ar("unknown", 0.50), _loc())
        assert decision.send_drone is False

    def test_unknown_high_confidence(self) -> None:
        decision = decide(_ar("unknown", 0.90), _loc())
        assert decision.send_drone is True
        assert decision.send_lora is True
        assert decision.priority == "low"


# ---------------------------------------------------------------------------
# PRIORITY_MAP structure
# ---------------------------------------------------------------------------


class TestPriorityMap:
    def test_priority_map_has_all_classes(self) -> None:
        expected_keys = {
            "chainsaw",
            "gunshot",
            "engine",
            "axe",
            "fire",
            "background",
            "unknown",
        }
        assert set(PRIORITY_MAP.keys()) == expected_keys

    def test_no_birds_in_priority_map(self) -> None:
        assert "birds" not in PRIORITY_MAP


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestDecisionOutput:
    def test_coordinates_in_reason(self) -> None:
        loc = _loc(lat=55.7510, lon=37.6130)
        decision = decide(_ar("chainsaw", 0.90), loc)
        assert "55.7510" in decision.reason
        assert "37.6130" in decision.reason

    def test_decision_dataclass_fields(self) -> None:
        d = Decision(send_drone=True, send_lora=True, priority="high", reason="test")
        assert hasattr(d, "send_drone")
        assert hasattr(d, "send_lora")
        assert hasattr(d, "priority")
        assert hasattr(d, "reason")
