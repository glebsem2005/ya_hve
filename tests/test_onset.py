"""Tests for edge.audio.onset — Sharp sound onset detector.

20 tests covering onset detection, sustained sound patterns,
stateful behavior, cooldown, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge.audio.onset import OnsetDetector, OnsetEvent, detect_onset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silence(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate silence (near-zero noise floor)."""
    return np.random.normal(0, 1e-5, int(sr * duration_s)).astype(np.float32)


def _loud_transient(
    duration_s: float = 0.5,
    sr: int = 16000,
    amplitude: float = 0.8,
    freq: float = 1000.0,
) -> np.ndarray:
    """Generate a loud sine burst simulating a sharp sound."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _chainsaw_like(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Simulate a chainsaw-like onset: silence followed by loud buzz."""
    quiet = _silence(duration_s * 0.3, sr)
    loud = _loud_transient(duration_s * 0.7, sr, amplitude=0.6, freq=500)
    return np.concatenate([quiet, loud])


def _gunshot_like(sr: int = 16000) -> np.ndarray:
    """Simulate a gunshot-like impulse: very short, very loud."""
    quiet = _silence(0.5, sr)
    impulse = np.random.normal(0, 0.9, int(sr * 0.05)).astype(np.float32)
    tail = _silence(0.5, sr)
    return np.concatenate([quiet, impulse, tail])


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------


class TestOnsetDetection:
    def test_silence_no_trigger(self) -> None:
        """Pure silence should not trigger onset."""
        waveform = _silence(2.0)
        event = detect_onset(waveform)
        assert not event.triggered

    def test_loud_transient_triggers(self) -> None:
        """A sudden loud sound after silence should trigger."""
        waveform = _chainsaw_like(2.0)
        event = detect_onset(waveform)
        assert event.triggered
        assert event.energy_ratio > 1.0

    def test_gunshot_triggers(self) -> None:
        """Short impulse (gunshot-like) should trigger."""
        waveform = _gunshot_like()
        event = detect_onset(waveform)
        assert event.triggered

    def test_constant_noise_no_trigger(self) -> None:
        """Constant moderate noise should NOT trigger (no transient).

        Key: the detector's baseline must match the noise level.
        If baseline is built from the same noise level, no spike occurs.
        """
        detector = OnsetDetector()
        noise_level = 0.1
        # Build baseline from same noise level
        baseline = np.random.normal(0, noise_level, 32000).astype(np.float32)
        detector.detect(baseline)
        # Feed more of the same — no sudden change
        waveform = np.random.normal(0, noise_level, 32000).astype(np.float32)
        event = detector.detect(waveform)
        assert not event.triggered

    def test_gradual_increase_no_trigger(self) -> None:
        """Slowly increasing volume should not trigger onset detection.

        We use a very high threshold so gradual ramps don't cross it.
        """
        n = 32000
        # Start from a non-trivial baseline so the ramp ratio stays low
        detector = OnsetDetector(energy_ratio_threshold=50.0)
        baseline = np.random.normal(0, 0.05, 16000).astype(np.float32)
        detector.detect(baseline)
        # Gradual ramp from 0.05 to 0.15 — ratio ~3x at most
        envelope = np.linspace(0.05, 0.15, n)
        noise = np.random.normal(0, 1, n).astype(np.float32)
        waveform = (noise * envelope).astype(np.float32)
        event = detector.detect(waveform)
        assert not event.triggered


# ---------------------------------------------------------------------------
# OnsetEvent dataclass
# ---------------------------------------------------------------------------


class TestOnsetEvent:
    def test_event_fields(self) -> None:
        event = OnsetEvent(
            triggered=True, frame_index=5, energy_ratio=12.5, peak_energy=0.3
        )
        assert event.triggered is True
        assert event.frame_index == 5
        assert event.energy_ratio == 12.5
        assert event.peak_energy == 0.3


# ---------------------------------------------------------------------------
# Stateful detector
# ---------------------------------------------------------------------------


class TestOnsetDetectorStateful:
    def test_detector_reset(self) -> None:
        """Reset clears internal state."""
        detector = OnsetDetector()
        detector.detect(_silence(1.0))
        assert len(detector._energy_history) > 0
        detector.reset()
        assert len(detector._energy_history) == 0
        assert detector._cooldown_counter == 0

    def test_cooldown_prevents_double_trigger(self) -> None:
        """After a trigger, cooldown should prevent immediate re-trigger."""
        detector = OnsetDetector(cooldown_frames=100)
        # Pre-fill baseline
        detector.detect(_silence(1.0))
        # First trigger
        event1 = detector.detect(_chainsaw_like(1.0))
        assert event1.triggered
        # Immediate second signal — should be suppressed by cooldown
        event2 = detector.detect(_chainsaw_like(0.5))
        assert not event2.triggered

    def test_sequential_chunks(self) -> None:
        """Detector accumulates baseline across multiple calls."""
        detector = OnsetDetector()
        # Feed several silence chunks to build baseline
        for _ in range(5):
            event = detector.detect(_silence(0.2))
            assert not event.triggered
        # Now send a loud chunk
        event = detector.detect(_chainsaw_like(1.0))
        assert event.triggered

    def test_custom_threshold(self) -> None:
        """Higher threshold requires bigger spikes."""
        detector_low = OnsetDetector(energy_ratio_threshold=3.0)
        detector_high = OnsetDetector(energy_ratio_threshold=50.0)

        # Build baseline for both
        baseline = _silence(1.0)
        detector_low.detect(baseline.copy())
        detector_high.detect(baseline.copy())

        waveform = _chainsaw_like(1.0)
        event_low = detector_low.detect(waveform.copy())
        event_high = detector_high.detect(waveform.copy())

        assert event_low.triggered
        # Very high threshold may not trigger on moderate chainsaw
        # (depends on exact ratio — this test validates threshold comparison)
        assert event_low.energy_ratio < 50.0 or event_high.triggered


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestOnsetEdgeCases:
    def test_empty_waveform(self) -> None:
        """Empty input should return non-triggered event."""
        event = detect_onset(np.array([], dtype=np.float32))
        assert not event.triggered

    def test_very_short_waveform(self) -> None:
        """Waveform shorter than frame_size should not crash."""
        waveform = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        event = detect_onset(waveform)
        assert not event.triggered

    def test_single_sample_spike(self) -> None:
        """A single very loud sample in silence should still work."""
        waveform = _silence(1.0)
        waveform[8000] = 1.0  # single spike
        event = detect_onset(waveform)
        # May or may not trigger depending on frame averaging
        assert isinstance(event, OnsetEvent)

    def test_detect_onset_convenience(self) -> None:
        """Stateless convenience function should work."""
        waveform = _gunshot_like()
        event = detect_onset(waveform)
        assert isinstance(event, OnsetEvent)
        assert event.triggered


# ---------------------------------------------------------------------------
# Sustained sound onset (silence → long continuous sound)
# ---------------------------------------------------------------------------


def _sustained_chainsaw(
    quiet_s: float = 1.0, loud_s: float = 4.0, sr: int = 16000
) -> np.ndarray:
    """Mimic demo/generate_audio.py chainsaw: quiet ambience → sustained buzz."""
    rng = np.random.default_rng(42)
    quiet = rng.normal(0, 0.005, int(sr * quiet_s)).astype(np.float32)

    n_loud = int(sr * loud_s)
    t = np.linspace(0, loud_s, n_loud, endpoint=False)
    freq = 100 + 300 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    phase = np.cumsum(freq / sr)
    saw = 2.0 * (phase % 1.0) - 1.0
    noise = rng.normal(0, 0.3, n_loud)
    chainsaw = (0.6 * saw + 0.4 * noise).astype(np.float32)

    fade_len = int(0.05 * sr)
    chainsaw[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)

    return np.concatenate([quiet, chainsaw])


def _sustained_engine(
    quiet_s: float = 1.0, loud_s: float = 4.0, sr: int = 16000
) -> np.ndarray:
    """Quiet ambience → sustained engine hum (low-freq sine + harmonics)."""
    rng = np.random.default_rng(99)
    quiet = rng.normal(0, 0.005, int(sr * quiet_s)).astype(np.float32)

    n_loud = int(sr * loud_s)
    t = np.linspace(0, loud_s, n_loud, endpoint=False)
    engine = (
        0.4 * np.sin(2 * np.pi * 80 * t)
        + 0.3 * np.sin(2 * np.pi * 160 * t)
        + 0.2 * np.sin(2 * np.pi * 240 * t)
    ).astype(np.float32)
    engine += rng.normal(0, 0.1, n_loud).astype(np.float32)

    fade_len = int(0.05 * sr)
    engine[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)

    return np.concatenate([quiet, engine])


class TestSustainedOnsetDetection:
    """Tests for the pattern: quiet ambience → sustained loud sound.

    This is the real-world pattern for chainsaw/engine — they start
    abruptly and run continuously, unlike a gunshot impulse.
    """

    def test_silence_then_sustained_chainsaw_triggers(self) -> None:
        """1s quiet → 4s chainsaw must trigger onset with ratio >= 8.0."""
        waveform = _sustained_chainsaw()
        event = detect_onset(waveform)
        assert event.triggered
        assert event.energy_ratio >= 8.0

    def test_silence_then_sustained_engine_triggers(self) -> None:
        """1s quiet → 4s engine hum must trigger onset."""
        waveform = _sustained_engine()
        event = detect_onset(waveform)
        assert event.triggered

    def test_sustained_sound_without_quiet_lead_no_trigger(self) -> None:
        """Chainsaw from sample 0 (no quiet lead-in) should NOT trigger.

        This reproduces the original bug: generate_chainsaw() had no
        silence prefix, so the baseline adapted to the chainsaw level
        and ratio stayed ~2.2 (below threshold 8.0).

        Uses OnsetDetector directly (no zero pre-fill) to match how
        the edge server processes audio from MicSimulator.
        """
        rng = np.random.default_rng(42)
        n = 16000 * 5
        t = np.linspace(0, 5.0, n, endpoint=False)
        freq = 100 + 300 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        phase = np.cumsum(freq / 16000)
        saw = 2.0 * (phase % 1.0) - 1.0
        noise = rng.normal(0, 0.3, n)
        mixed = np.clip(0.6 * saw + 0.4 * noise, -1.0, 1.0).astype(np.float32)

        detector = OnsetDetector()
        event = detector.detect(mixed)
        assert not event.triggered

    def test_demo_chainsaw_pattern_matches_threshold(self) -> None:
        """Exact pattern from generate_chainsaw() must pass onset threshold.

        Regression guard: if generate_audio.py changes, this test
        ensures the onset detector still sees a sharp enough transition.
        """
        waveform = _sustained_chainsaw(quiet_s=1.0, loud_s=4.0)
        event = detect_onset(waveform)
        assert event.triggered, (
            f"Demo chainsaw pattern failed onset detection "
            f"(ratio={event.energy_ratio:.1f}, threshold=8.0)"
        )
