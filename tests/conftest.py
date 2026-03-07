"""Shared fixtures for acoustic forest monitoring tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from edge.audio.classifier import AudioResult, AudioClass
from edge.tdoa.triangulate import MicPosition, TriangulationResult


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate used across the project."""
    return 16000


# ---------------------------------------------------------------------------
# YAMNet mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_yamnet_model() -> MagicMock:
    """MagicMock that behaves like TF-Hub YAMNet.

    Returns:
        A callable mock producing (scores, embeddings, spectrogram) where
        embeddings.numpy() yields shape (5, 1024).
    """
    yamnet = MagicMock(name="yamnet")

    scores = MagicMock(name="scores")
    scores.numpy.return_value = np.random.rand(5, 521).astype(np.float32)

    embeddings = MagicMock(name="embeddings")
    embeddings.numpy.return_value = np.random.randn(5, 1024).astype(np.float32)

    spectrogram = MagicMock(name="spectrogram")
    spectrogram.numpy.return_value = np.random.rand(5, 64).astype(np.float32)

    yamnet.return_value = (scores, embeddings, spectrogram)
    return yamnet


@pytest.fixture
def mock_head_model() -> MagicMock:
    """MagicMock that behaves like the Keras classification head.

    input_shape is (None, 2181) and __call__ returns a 6-class softmax-like
    prediction with class 0 (chainsaw) having the highest score by default.
    """
    head = MagicMock(name="head")
    head.input_shape = (None, 2181)

    default_pred = np.array([[0.60, 0.10, 0.08, 0.07, 0.05, 0.10]], dtype=np.float32)
    pred_tensor = MagicMock(name="pred_tensor")
    pred_tensor.numpy.return_value = default_pred
    head.return_value = pred_tensor

    return head


# ---------------------------------------------------------------------------
# Dataclass factories
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_result_factory() -> Callable[..., AudioResult]:
    """Factory that builds an AudioResult with sensible defaults."""

    def _make(
        label: AudioClass = "chainsaw",
        confidence: float = 0.90,
        raw_scores: dict | None = None,
    ) -> AudioResult:
        if raw_scores is None:
            raw_scores = {
                "chainsaw": 0.0,
                "gunshot": 0.0,
                "engine": 0.0,
                "axe": 0.0,
                "fire": 0.0,
                "background": 0.0,
            }
            raw_scores[label] = confidence
        return AudioResult(label=label, confidence=confidence, raw_scores=raw_scores)

    return _make


@pytest.fixture
def triangulation_result_factory() -> Callable[..., TriangulationResult]:
    """Factory that builds a TriangulationResult with sensible defaults."""

    def _make(
        lat: float = 55.7510,
        lon: float = 37.6130,
        error_m: float = 5.0,
    ) -> TriangulationResult:
        return TriangulationResult(lat=lat, lon=lon, error_m=error_m)

    return _make


# ---------------------------------------------------------------------------
# Microphone geometry
# ---------------------------------------------------------------------------


@pytest.fixture
def triangle_mics() -> list[MicPosition]:
    """Three microphones forming a roughly equilateral triangle (~100 m sides).

    Centred near 55.751 N, 37.613 E (Moscow).  Offsets are approximately
    100 m in latitude and longitude directions.
    """
    # ~100 m offsets at this latitude:
    # 1 degree lat ~ 111 320 m  =>  100 m ~ 0.000898 deg
    # 1 degree lon ~ 63 600 m   =>  100 m ~ 0.001572 deg (at lat 55.75)
    mic_a = MicPosition(lat=55.7510, lon=37.6130)
    mic_b = MicPosition(lat=55.7510, lon=37.6146)  # ~100 m east
    mic_c = MicPosition(lat=55.7519, lon=37.6138)  # ~100 m north
    return [mic_a, mic_b, mic_c]
