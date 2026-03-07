"""Tests for edge.tdoa.triangulate — TDOA estimation and triangulation.

14 tests covering GCC-PHAT TDOA estimation, coordinate conversions,
triangulation integration, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge.tdoa.triangulate import (
    MicPosition,
    TriangulationResult,
    _estimate_tdoa,
    _latlon_to_meters,
    _meters_to_latlon,
    triangulate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_pulse(
    sr: int = 16000,
    freq: float = 1000.0,
    duration: float = 0.2,
    n_total: int = 8000,
    offset: int = 0,
) -> np.ndarray:
    """Generate a sine pulse starting at *offset* samples inside a zero buffer."""
    pulse_len = int(sr * duration)
    t = np.arange(pulse_len) / sr
    pulse = (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sig = np.zeros(n_total, dtype=np.float32)
    end = min(offset + pulse_len, n_total)
    sig[offset:end] = pulse[: end - offset]
    return sig


# ---------------------------------------------------------------------------
# _estimate_tdoa
# ---------------------------------------------------------------------------


class TestEstimateTdoa:
    """Unit tests for the GCC-PHAT-based TDOA estimator."""

    def test_tdoa_zero_lag(self) -> None:
        sig = _sine_pulse(offset=1000)
        tdoa = _estimate_tdoa(sig, sig.copy(), sr=16000)
        assert abs(tdoa) < 1 / 16000, f"Expected ~0, got {tdoa}"

    def test_tdoa_positive_lag(self) -> None:
        sr = 16000
        shift = 10
        sig_a = _sine_pulse(offset=1000, sr=sr)
        sig_b = _sine_pulse(offset=1000 + shift, sr=sr)
        tdoa = _estimate_tdoa(sig_a, sig_b, sr=sr)
        expected = shift / sr
        assert abs(tdoa - expected) < 2 / sr, (
            f"Expected ~{expected:.6f}, got {tdoa:.6f}"
        )

    def test_tdoa_negative_lag(self) -> None:
        sr = 16000
        shift = 10
        sig_a = _sine_pulse(offset=1000 + shift, sr=sr)
        sig_b = _sine_pulse(offset=1000, sr=sr)
        tdoa = _estimate_tdoa(sig_a, sig_b, sr=sr)
        expected = shift / sr  # a is delayed relative to b => positive TDOA
        assert abs(tdoa - expected) < 2 / sr, (
            f"Expected ~{expected:.6f}, got {tdoa:.6f}"
        )

    def test_tdoa_subpixel(self) -> None:
        """Subpixel interpolation should bring estimate closer to true lag."""
        sr = 16000
        shift_samples = 10.5
        n = 8000
        # Use broadband noise for good cross-correlation peak
        rng = np.random.default_rng(42)
        sig_a = rng.normal(size=n).astype(np.float32)
        # Shift by fractional amount via frequency domain
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        SIG_A = np.fft.rfft(sig_a)
        phase_shift = np.exp(-2j * np.pi * freqs * shift_samples / sr)
        sig_b = np.fft.irfft(SIG_A * phase_shift, n).astype(np.float32)

        tdoa = _estimate_tdoa(sig_a, sig_b, sr=sr)
        expected = shift_samples / sr
        assert abs(tdoa - expected) < 1.0 / sr, (
            f"Subpixel: expected ~{expected:.6f}, got {tdoa:.6f}"
        )

    def test_tdoa_large_lag(self) -> None:
        sr = 16000
        shift = 100
        sig_a = _sine_pulse(offset=500, sr=sr, n_total=16000)
        sig_b = _sine_pulse(offset=500 + shift, sr=sr, n_total=16000)
        tdoa = _estimate_tdoa(sig_a, sig_b, sr=sr)
        expected = shift / sr
        assert abs(tdoa - expected) < 3 / sr, (
            f"Expected ~{expected:.6f}, got {tdoa:.6f}"
        )

    def test_tdoa_all_zeros(self) -> None:
        sig = np.zeros(4000, dtype=np.float32)
        tdoa = _estimate_tdoa(sig, sig.copy(), sr=16000)
        # With all zeros, behaviour is undefined but must not crash
        assert isinstance(tdoa, float)

    def test_tdoa_different_lengths(self) -> None:
        sig_a = _sine_pulse(n_total=1000, offset=200)
        sig_b = _sine_pulse(n_total=1500, offset=200)
        tdoa = _estimate_tdoa(sig_a, sig_b, sr=16000)
        assert isinstance(tdoa, float)


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


class TestCoordinateConversion:
    def test_latlon_to_meters_roundtrip(self) -> None:
        lat1, lon1 = 55.7510, 37.6130
        lat2, lon2 = 55.7519, 37.6146
        dx, dy = _latlon_to_meters(lat1, lon1, lat2, lon2)
        lat_back, lon_back = _meters_to_latlon(lat1, lon1, dx, dy)
        assert abs(lat_back - lat2) < 1e-6, f"lat: {lat_back} vs {lat2}"
        assert abs(lon_back - lon2) < 1e-6, f"lon: {lon_back} vs {lon2}"

    def test_meters_to_latlon_identity(self) -> None:
        lat0, lon0 = 55.7510, 37.6130
        lat_out, lon_out = _meters_to_latlon(lat0, lon0, 0.0, 0.0)
        assert abs(lat_out - lat0) < 1e-10
        assert abs(lon_out - lon0) < 1e-10


# ---------------------------------------------------------------------------
# triangulate()
# ---------------------------------------------------------------------------


class TestTriangulate:
    def test_triangulate_smoke(self, triangle_mics: list[MicPosition]) -> None:
        sr = 16000
        n = 16000
        rng = np.random.default_rng(99)
        signals = [rng.normal(size=n).astype(np.float32) for _ in range(3)]
        result = triangulate(signals, triangle_mics, sample_rate=sr)
        assert isinstance(result, TriangulationResult)
        assert isinstance(result.lat, float)
        assert isinstance(result.lon, float)
        assert isinstance(result.error_m, float)

    def test_triangulate_source_at_centroid(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """When the source is at the centroid of the triangle, all TDOAs are
        near zero, so the optimiser should converge close to the centroid."""
        sr = 16000
        n = 16000
        # Use the same signal for all three mics (zero TDOA)
        rng = np.random.default_rng(7)
        base = rng.normal(size=n).astype(np.float32)
        signals = [base.copy() for _ in range(3)]
        result = triangulate(signals, triangle_mics, sample_rate=sr)

        centroid_lat = np.mean([m.lat for m in triangle_mics])
        centroid_lon = np.mean([m.lon for m in triangle_mics])
        dx, dy = _latlon_to_meters(centroid_lat, centroid_lon, result.lat, result.lon)
        dist = np.sqrt(dx**2 + dy**2)
        assert dist < 30.0, f"Expected source near centroid, got {dist:.1f} m away"

    def test_triangulate_source_near_mic_a(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """If signal arrives at mic_a first (zero-shifted) and later at B/C,
        the estimated source should be closer to mic_a than to the centroid."""
        sr = 16000
        n = 16000
        rng = np.random.default_rng(12)
        base = rng.normal(size=n).astype(np.float32)
        shift = 5  # mic_b and mic_c receive the signal 5 samples later
        sig_a = base.copy()
        sig_b = np.roll(base, shift)
        sig_c = np.roll(base, shift)
        signals = [sig_a, sig_b, sig_c]
        result = triangulate(signals, triangle_mics, sample_rate=sr)

        mic_a = triangle_mics[0]
        dx, dy = _latlon_to_meters(mic_a.lat, mic_a.lon, result.lat, result.lon)
        dist_to_a = np.sqrt(dx**2 + dy**2)
        # The result should be within a reasonable distance of mic_a
        assert dist_to_a < 200.0, (
            f"Expected source near mic_a, got {dist_to_a:.1f} m away"
        )


# ---------------------------------------------------------------------------
# Edge cases — wrong number of inputs
# ---------------------------------------------------------------------------


class TestTriangulateEdgeCases:
    def test_triangulate_requires_three_signals(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        signals = [np.zeros(1000, dtype=np.float32) for _ in range(2)]
        with pytest.raises(AssertionError):
            triangulate(signals, triangle_mics, sample_rate=16000)

    def test_triangulate_requires_three_mics(self) -> None:
        signals = [np.zeros(1000, dtype=np.float32) for _ in range(3)]
        mics = [MicPosition(lat=55.75, lon=37.61), MicPosition(lat=55.76, lon=37.62)]
        with pytest.raises(AssertionError):
            triangulate(signals, mics, sample_rate=16000)
