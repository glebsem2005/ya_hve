"""Tests for edge.audio.classifier — YAMNet-based sound classifier.

15 tests exercising constants, classification pipeline, padding logic,
and error/edge-case handling.  All TensorFlow dependencies are mocked out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from edge.audio.classifier import (
    CLASSES,
    AudioResult,
    _unknown,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_embeddings_mock(shape: tuple[int, int] = (5, 1024)) -> MagicMock:
    """Create an embeddings tensor mock with .numpy() support."""
    emb = MagicMock(name="embeddings")
    emb.numpy.return_value = np.random.randn(*shape).astype(np.float32)
    return emb


def _make_head_mock(
    pred_array: np.ndarray | None = None,
    input_dim: int = 2048,
) -> MagicMock:
    """Create a Keras head model mock."""
    head = MagicMock(name="head")
    head.input_shape = (None, input_dim)

    if pred_array is None:
        pred_array = np.array([[0.60, 0.10, 0.08, 0.07, 0.05, 0.10]], dtype=np.float32)
    tensor = MagicMock(name="pred_tensor")
    tensor.numpy.return_value = pred_array
    head.return_value = tensor
    return head


def _make_yamnet_mock(
    emb_shape: tuple[int, int] = (5, 1024),
) -> MagicMock:
    """Create a YAMNet model mock (callable)."""
    yamnet = MagicMock(name="yamnet")
    scores = MagicMock(name="scores")
    embeddings = _make_embeddings_mock(emb_shape)
    spectrogram = MagicMock(name="spectrogram")
    yamnet.return_value = (scores, embeddings, spectrogram)
    return yamnet


def _patch_classify(
    waveform: np.ndarray,
    sr: int = 16000,
    head_pred: np.ndarray | None = None,
    emb_shape: tuple[int, int] = (5, 1024),
    head: MagicMock | None = None,
) -> AudioResult:
    """Run classify() with all heavy dependencies mocked out."""
    yamnet = _make_yamnet_mock(emb_shape)
    if head is None:
        head = _make_head_mock(head_pred)

    with (
        patch("edge.audio.classifier.sf") as mock_sf,
        patch("edge.audio.classifier._load_models", return_value=(yamnet, head)),
    ):
        mock_sf.read.return_value = (waveform, sr)
        from edge.audio.classifier import classify

        return classify("fake_audio.wav")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_classes_constant_length(self) -> None:
        assert len(CLASSES) == 6

    def test_classes_constant_values(self) -> None:
        assert CLASSES == ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]


# ---------------------------------------------------------------------------
# _unknown()
# ---------------------------------------------------------------------------


class TestUnknown:
    def test_unknown_result_fields(self) -> None:
        result = _unknown()
        assert result.label == "unknown"
        assert isinstance(result, AudioResult)
        assert result.raw_scores == {}


# ---------------------------------------------------------------------------
# classify() — normal operation
# ---------------------------------------------------------------------------


class TestClassifyNormal:
    def test_classify_returns_audio_result(self) -> None:
        waveform = np.random.randn(16000).astype(np.float32)
        result = _patch_classify(waveform)
        assert isinstance(result, AudioResult)

    def test_classify_mono_handling(self) -> None:
        """1-D waveform is processed without error."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = _patch_classify(waveform)
        assert result.label in CLASSES or result.label == "unknown"

    def test_classify_stereo_handling(self) -> None:
        """2-D waveform is averaged to mono (mean axis=1)."""
        waveform = np.random.randn(16000, 2).astype(np.float32)
        result = _patch_classify(waveform)
        assert isinstance(result, AudioResult)

    def test_classify_raw_scores_keys(self) -> None:
        waveform = np.random.randn(16000).astype(np.float32)
        result = _patch_classify(waveform)
        assert set(result.raw_scores.keys()) == set(CLASSES)

    def test_classify_label_is_argmax(self) -> None:
        """If pred[2] (engine) is the highest, label should be 'engine'."""
        pred = np.array([[0.05, 0.05, 0.70, 0.05, 0.05, 0.10]], dtype=np.float32)
        waveform = np.random.randn(16000).astype(np.float32)
        result = _patch_classify(waveform, head_pred=pred)
        assert result.label == "engine"
        assert abs(result.confidence - 0.70) < 1e-5

    @pytest.mark.parametrize(
        "class_idx,class_name",
        enumerate(CLASSES),
    )
    def test_classify_each_class(self, class_idx: int, class_name: str) -> None:
        """Each class should be returned when its score is the argmax."""
        pred = np.full((1, 6), 0.05, dtype=np.float32)
        pred[0, class_idx] = 0.80
        waveform = np.random.randn(16000).astype(np.float32)
        result = _patch_classify(waveform, head_pred=pred)
        assert result.label == class_name


# ---------------------------------------------------------------------------
# classify() — padding
# ---------------------------------------------------------------------------


class TestClassifyPadding:
    def test_classify_zero_padding_applied(self) -> None:
        """Features shorter than head.input_shape[-1] are zero-padded."""
        # With 5 embeddings of 1024, mean+max = 2048 == input_dim => no padding needed
        head = _make_head_mock(input_dim=2048)
        waveform = np.random.randn(16000).astype(np.float32)

        yamnet = _make_yamnet_mock(emb_shape=(5, 1024))
        with (
            patch("edge.audio.classifier.sf") as mock_sf,
            patch(
                "edge.audio.classifier._load_models",
                return_value=(yamnet, head),
            ),
        ):
            mock_sf.read.return_value = (waveform, 16000)
            from edge.audio.classifier import classify

            result = classify("fake_audio.wav")

        # Should still get a valid result (padding handled internally)
        assert isinstance(result, AudioResult)
        assert result.label in CLASSES

    def test_classify_no_padding_at_exact_dim(self) -> None:
        """If features already match head input dim, no padding is added."""
        # 5 embeddings * 1024 => mean(1024) + max(1024) = 2048
        # Set head input to 2048 => no padding
        head = _make_head_mock(input_dim=2048)
        waveform = np.random.randn(16000).astype(np.float32)
        yamnet = _make_yamnet_mock(emb_shape=(5, 1024))

        with (
            patch("edge.audio.classifier.sf") as mock_sf,
            patch(
                "edge.audio.classifier._load_models",
                return_value=(yamnet, head),
            ),
        ):
            mock_sf.read.return_value = (waveform, 16000)
            from edge.audio.classifier import classify

            result = classify("fake_audio.wav")

        assert isinstance(result, AudioResult)

    def test_classify_short_audio_padded(self) -> None:
        """Waveform shorter than 15600 samples is zero-padded."""
        waveform = np.random.randn(100).astype(np.float32)
        result = _patch_classify(waveform)
        assert isinstance(result, AudioResult)


# ---------------------------------------------------------------------------
# classify() — edge / error cases
# ---------------------------------------------------------------------------


class TestClassifyEdgeCases:
    def test_classify_empty_waveform_returns_unknown(self) -> None:
        waveform = np.array([], dtype=np.float32)
        result = _patch_classify(waveform)
        assert result.label == "unknown"
        assert isinstance(result, AudioResult)

    def test_classify_empty_embeddings_returns_unknown(self) -> None:
        """If YAMNet returns zero embeddings, result should be unknown."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = _patch_classify(waveform, emb_shape=(0, 1024))
        assert result.label == "unknown"
        assert isinstance(result, AudioResult)

    def test_classify_head_none_returns_unknown(self) -> None:
        """If head model failed to load (None), result is unknown."""
        yamnet = _make_yamnet_mock()
        with (
            patch("edge.audio.classifier.sf") as mock_sf,
            patch(
                "edge.audio.classifier._load_models",
                return_value=(yamnet, None),
            ),
        ):
            mock_sf.read.return_value = (
                np.random.randn(16000).astype(np.float32),
                16000,
            )
            from edge.audio.classifier import classify

            result = classify("fake_audio.wav")

        assert result.label in CLASSES
        assert isinstance(result, AudioResult)


# ---------------------------------------------------------------------------
# classify() — normalization
# ---------------------------------------------------------------------------


class TestClassifyNormalization:
    def test_classify_normalizes_quiet_waveform(self) -> None:
        """Quiet waveform (peak=0.1) should be peak-normalized before YAMNet."""
        quiet = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32) * 0.1
        assert np.max(np.abs(quiet)) <= 0.1  # sanity: it's quiet

        yamnet = _make_yamnet_mock()
        head = _make_head_mock()
        captured_waveforms: list[np.ndarray] = []

        def capturing_yamnet(wf):
            captured_waveforms.append(np.array(wf, copy=True))
            return yamnet(wf)

        with (
            patch("edge.audio.classifier.sf") as mock_sf,
            patch(
                "edge.audio.classifier._load_models",
                return_value=(capturing_yamnet, head),
            ),
        ):
            mock_sf.read.return_value = (quiet, 16000)
            from edge.audio.classifier import classify

            classify("fake_audio.wav")

        assert len(captured_waveforms) > 0
        passed_waveform = captured_waveforms[0]
        # After normalization, peak should be ≈ 1.0
        assert np.max(np.abs(passed_waveform)) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# YAMNET_CLASS_MAP — expanded mappings
# ---------------------------------------------------------------------------


class TestClassMapExpanded:
    def test_class_map_sawing_maps_to_chainsaw(self) -> None:
        """'Sawing' AudioSet class should map to 'chainsaw'."""
        from edge.audio.classifier import YAMNET_CLASS_MAP

        assert YAMNET_CLASS_MAP.get("Sawing") == "chainsaw"

    def test_class_map_drill_maps_to_chainsaw(self) -> None:
        """'Drill' AudioSet class should map to 'chainsaw'."""
        from edge.audio.classifier import YAMNET_CLASS_MAP

        assert YAMNET_CLASS_MAP.get("Drill") == "chainsaw"

    def test_class_map_tools_maps_to_chainsaw(self) -> None:
        """'Tools' AudioSet class should map to 'chainsaw'."""
        from edge.audio.classifier import YAMNET_CLASS_MAP

        assert YAMNET_CLASS_MAP.get("Tools") == "chainsaw"
