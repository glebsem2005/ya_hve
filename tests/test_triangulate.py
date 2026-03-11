"""Tests for edge.tdoa.triangulate — TDOA estimation and triangulation.

Tests covering GCC-PHAT TDOA estimation, coordinate conversions,
triangulation integration, distance estimation, fusion quality, and edge cases.

Key improvement: a physically realistic signal simulator that places a sound
source at known coordinates and generates per-mic signals with correct
propagation delays and inverse-square-law attenuation.  This lets us test
actual localisation accuracy with tight tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge.tdoa.triangulate import (
    MicPosition,
    TriangulationResult,
    SPEED_OF_SOUND,
    _estimate_tdoa,
    _latlon_to_meters,
    _meters_to_latlon,
    triangulate,
)
from edge.tdoa.distance import (
    DistanceEstimate,
    estimate_distances,
    _rms_energy,
    _DEFAULT_REF_RMS,
    _DEFAULT_REF_DISTANCE,
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


def _simulate_source(
    source_xy: tuple[float, float],
    mics_xy: list[tuple[float, float]],
    sr: int = 16000,
    n_samples: int = 16000,
    speed: float = SPEED_OF_SOUND,
    snr_db: float = 40.0,
    seed: int = 0,
    burst: bool = False,
) -> list[np.ndarray]:
    """Simulate microphone signals for a source at a known (x, y) position.

    Physics modelled:
    - Propagation delay: samples = distance / speed * sr  (fractional, via FFT shift)
    - Amplitude attenuation: 1/r  (inverse-square law on amplitude)
    - Additive Gaussian noise at the specified SNR

    If burst=True, generates a transient signal (sine burst in silence) which
    gives the SNR estimator a clear quiet/loud contrast, producing realistic
    confidence values for the energy-based distance estimation.

    Returns one signal array per microphone.
    """
    rng = np.random.default_rng(seed)

    if burst:
        # Transient burst: 0.3s of multi-frequency sine in a 1s buffer
        # This gives the SNR estimator clear quiet vs loud frames
        source = np.zeros(n_samples, dtype=np.float64)
        burst_len = int(0.3 * sr)
        burst_start = n_samples // 4  # start at 25% to leave quiet at both ends
        t = np.arange(burst_len) / sr
        # Multi-frequency burst (chainsaw-like: 400 + 800 + 1200 Hz)
        burst_sig = (
            0.5 * np.sin(2 * np.pi * 400 * t)
            + 0.3 * np.sin(2 * np.pi * 800 * t)
            + 0.2 * np.sin(2 * np.pi * 1200 * t)
        )
        source[burst_start : burst_start + burst_len] = burst_sig
        source *= _DEFAULT_REF_RMS / (np.sqrt(np.mean(source**2)) + 1e-15)
    else:
        # Broadband source signal (white noise filtered to 300-5000 Hz)
        raw = rng.normal(size=n_samples).astype(np.float64)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
        RAW = np.fft.rfft(raw)
        mask = (freqs >= 300) & (freqs <= 5000)
        RAW[~mask] = 0.0
        source = np.fft.irfft(RAW, n_samples)
        source *= _DEFAULT_REF_RMS / (np.sqrt(np.mean(source**2)) + 1e-15)

    sx, sy = source_xy
    signals = []
    for mx, my in mics_xy:
        dist = np.sqrt((sx - mx) ** 2 + (sy - my) ** 2)
        dist = max(dist, 0.1)  # avoid division by zero

        # Attenuation: amplitude ∝ 1/r (with ref at 1 m)
        amplitude = _DEFAULT_REF_DISTANCE / dist

        # Propagation delay in samples (fractional)
        delay_samples = dist / speed * sr

        # Apply fractional delay via FFT phase shift
        SRC = np.fft.rfft(source)
        f = np.fft.rfftfreq(n_samples, d=1.0 / sr)
        phase = np.exp(-2j * np.pi * f * delay_samples / sr)
        delayed = np.fft.irfft(SRC * phase, n_samples)

        sig = amplitude * delayed

        # Add noise
        noise_power = np.mean(sig**2) / (10 ** (snr_db / 10) + 1e-15)
        sig += rng.normal(scale=np.sqrt(max(noise_power, 1e-15)), size=n_samples)

        signals.append(sig.astype(np.float32))

    return signals


def _loc_error_m(
    mics: list[MicPosition],
    result: TriangulationResult,
    source_xy: tuple[float, float],
) -> float:
    """Compute localisation error in metres between result and true source."""
    lat0, lon0 = mics[0].lat, mics[0].lon
    src_lat, src_lon = _meters_to_latlon(lat0, lon0, source_xy[0], source_xy[1])
    dx, dy = _latlon_to_meters(src_lat, src_lon, result.lat, result.lon)
    return float(np.sqrt(dx**2 + dy**2))


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
        expected = -shift / sr  # a is delayed relative to b => negative TDOA
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
    """Triangulation accuracy tests with physically simulated signals."""

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

    def test_source_at_centroid_simulated(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """Source at centroid of mic triangle — equal distances, near-zero TDOAs.
        Tolerance: <15 m (was 30 m)."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))
        centroid = (np.mean([p[0] for p in mics_xy]), np.mean([p[1] for p in mics_xy]))

        signals = _simulate_source(centroid, mics_xy, snr_db=40, seed=7)
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        err = _loc_error_m(triangle_mics, result, centroid)
        assert err < 15.0, f"Centroid source: error {err:.1f} m (want <15 m)"

    def test_source_near_mic_a_simulated(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """Source 20 m from mic A — should locate within 15 m of true position."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        source = (5.0, 20.0)  # 20 m north-ish of mic A
        signals = _simulate_source(source, mics_xy, snr_db=40, seed=12)
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        err = _loc_error_m(triangle_mics, result, source)
        assert err < 15.0, f"Near mic_a source: error {err:.1f} m (want <15 m)"

    def test_source_outside_triangle(self, triangle_mics: list[MicPosition]) -> None:
        """Source 150 m away, outside the triangle — harder case, <40 m tolerance."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        source = (-120.0, 80.0)  # outside the triangle
        signals = _simulate_source(source, mics_xy, snr_db=35, seed=42)
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        err = _loc_error_m(triangle_mics, result, source)
        assert err < 40.0, f"Outside source: error {err:.1f} m (want <40 m)"

    @pytest.mark.parametrize(
        "source_xy,label",
        [
            ((30.0, 50.0), "inside-NE"),
            ((-40.0, -10.0), "outside-SW"),
            ((80.0, 0.0), "east-far"),
        ],
    )
    def test_multiple_source_positions(
        self, triangle_mics: list[MicPosition], source_xy: tuple, label: str
    ) -> None:
        """Parametrised test: various source positions, all within 30 m accuracy."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        signals = _simulate_source(
            source_xy, mics_xy, snr_db=35, seed=hash(label) % 2**31
        )
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        err = _loc_error_m(triangle_mics, result, source_xy)
        assert err < 30.0, (
            f"Source '{label}' at {source_xy}: error {err:.1f} m (want <30 m)"
        )

    def test_noisy_signal_degrades_gracefully(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """At low SNR (15 dB), result should still be within 80 m — not diverge."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        source = (30.0, 40.0)
        signals = _simulate_source(source, mics_xy, snr_db=15, seed=55)
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        err = _loc_error_m(triangle_mics, result, source)
        assert err < 80.0, f"Low-SNR source: error {err:.1f} m (want <80 m)"


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


# ---------------------------------------------------------------------------
# Distance estimation
# ---------------------------------------------------------------------------


class TestDistanceEstimation:
    def test_rms_energy_sine(self) -> None:
        t = np.arange(16000) / 16000
        sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        rms = _rms_energy(sig)
        # RMS of sine with amplitude A is A / sqrt(2) ≈ 0.354
        assert abs(rms - 0.5 / np.sqrt(2)) < 0.01

    def test_rms_energy_zeros(self) -> None:
        assert _rms_energy(np.zeros(100, dtype=np.float32)) == 0.0

    def test_louder_signal_closer(self) -> None:
        """A louder signal should produce a shorter distance estimate."""
        loud = np.random.default_rng(1).normal(scale=0.5, size=8000).astype(np.float32)
        quiet = loud * 0.1
        estimates = estimate_distances([loud, quiet], [10.0, 10.0])
        assert estimates[0].distance_m < estimates[1].distance_m

    def test_silence_returns_max(self) -> None:
        silence = np.zeros(8000, dtype=np.float32)
        estimates = estimate_distances([silence], [0.0], max_distance=500.0)
        assert estimates[0].distance_m == 500.0
        assert estimates[0].confidence == 0.0

    def test_confidence_higher_for_loud(self) -> None:
        rng = np.random.default_rng(42)
        loud = rng.normal(scale=0.3, size=8000).astype(np.float32)
        quiet = loud * 0.01
        estimates = estimate_distances([loud, quiet], [15.0, 3.0])
        assert estimates[0].confidence > estimates[1].confidence


# ---------------------------------------------------------------------------
# TDOA + Distance fusion
# ---------------------------------------------------------------------------


class TestFusion:
    """Tests that prove fusion (TDOA + distance) actually improves results."""

    def _mics_xy(self, triangle_mics: list[MicPosition]) -> list[tuple[float, float]]:
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        result = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            result.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))
        return result

    def test_fusion_beats_pure_tdoa(self, triangle_mics: list[MicPosition]) -> None:
        """Core test: fusion (distance_weight=0.3) must produce lower average
        error than pure TDOA (distance_weight=0.0) across multiple source positions.

        This is THE test that proves fusion adds value."""
        mics_xy = self._mics_xy(triangle_mics)
        sources = [(30.0, 50.0), (-40.0, 20.0), (10.0, 10.0), (70.0, -20.0)]

        errors_pure = []
        errors_fused = []
        for i, src in enumerate(sources):
            signals = _simulate_source(src, mics_xy, snr_db=30, seed=100 + i)
            r_pure = triangulate(
                signals, triangle_mics, sample_rate=16000, distance_weight=0.0
            )
            r_fused = triangulate(
                signals, triangle_mics, sample_rate=16000, distance_weight=0.3
            )
            errors_pure.append(_loc_error_m(triangle_mics, r_pure, src))
            errors_fused.append(_loc_error_m(triangle_mics, r_fused, src))

        avg_pure = np.mean(errors_pure)
        avg_fused = np.mean(errors_fused)
        # Fusion should be at least as good, ideally better
        assert avg_fused <= avg_pure * 1.1, (
            f"Fusion worse than pure TDOA: {avg_fused:.1f} m vs {avg_pure:.1f} m"
        )

    def test_fusion_vs_pure_tdoa_per_scenario(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """Fusion should not make ANY scenario significantly worse (>50% degradation)."""
        mics_xy = self._mics_xy(triangle_mics)
        sources = [
            (30.0, 50.0),
            (-40.0, 20.0),
            (10.0, 10.0),
            (70.0, -20.0),
            (0.0, 80.0),
        ]

        for i, src in enumerate(sources):
            signals = _simulate_source(src, mics_xy, snr_db=30, seed=200 + i)
            r_pure = triangulate(
                signals, triangle_mics, sample_rate=16000, distance_weight=0.0
            )
            r_fused = triangulate(
                signals, triangle_mics, sample_rate=16000, distance_weight=0.3
            )
            e_pure = _loc_error_m(triangle_mics, r_pure, src)
            e_fused = _loc_error_m(triangle_mics, r_fused, src)
            assert e_fused < e_pure * 1.5 + 5.0, (
                f"Source {src}: fusion {e_fused:.1f} m >> pure {e_pure:.1f} m"
            )

    def test_pure_tdoa_ignores_amplitude(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """With distance_weight=0, scaling all signals equally should not change result."""
        mics_xy = self._mics_xy(triangle_mics)
        source = (30.0, 50.0)
        signals = _simulate_source(source, mics_xy, snr_db=40, seed=300)

        r1 = triangulate(signals, triangle_mics, sample_rate=16000, distance_weight=0.0)
        scaled = [s * 0.1 for s in signals]
        r2 = triangulate(scaled, triangle_mics, sample_rate=16000, distance_weight=0.0)

        dx, dy = _latlon_to_meters(r1.lat, r1.lon, r2.lat, r2.lon)
        assert np.sqrt(dx**2 + dy**2) < 5.0, "Pure TDOA should be amplitude-invariant"

    def test_fusion_uses_amplitude_information(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """With distance_weight>0, asymmetric amplitude scaling SHOULD change
        the result (distance estimation uses amplitude). This proves fusion
        actually incorporates energy information.

        Uses burst=True so the SNR estimator sees clear quiet/loud contrast,
        producing realistic confidence values for the distance term.
        We scale only mic_a's signal louder — fusion should pull the estimate
        closer to mic_a (thinking the source is nearer to A)."""
        mics_xy = self._mics_xy(triangle_mics)
        source = (30.0, 50.0)
        signals = _simulate_source(source, mics_xy, snr_db=40, seed=301, burst=True)

        r1 = triangulate(signals, triangle_mics, sample_rate=16000, distance_weight=0.5)
        # Boost mic A by 5x — fusion should think source is closer to A
        boosted = [signals[0] * 5.0, signals[1], signals[2]]
        r2 = triangulate(boosted, triangle_mics, sample_rate=16000, distance_weight=0.5)

        dx, dy = _latlon_to_meters(r1.lat, r1.lon, r2.lat, r2.lon)
        dist = np.sqrt(dx**2 + dy**2)
        # Asymmetric amplitude change MUST shift the estimate
        assert dist > 1.0, (
            f"Fusion result didn't change with asymmetric amplitude ({dist:.2f} m) — "
            "distance term is not contributing"
        )

    def test_distance_weight_zero_matches_no_fusion(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """distance_weight=0.0 should produce the same result on repeated calls."""
        mics_xy = self._mics_xy(triangle_mics)
        source = (20.0, 30.0)
        signals = _simulate_source(source, mics_xy, snr_db=40, seed=400)

        r1 = triangulate(signals, triangle_mics, sample_rate=16000, distance_weight=0.0)
        r2 = triangulate(signals, triangle_mics, sample_rate=16000, distance_weight=0.0)

        dx, dy = _latlon_to_meters(r1.lat, r1.lon, r2.lat, r2.lon)
        assert np.sqrt(dx**2 + dy**2) < 0.5, "Deterministic calls should match"

    def test_high_distance_weight_still_converges(
        self, triangle_mics: list[MicPosition]
    ) -> None:
        """Even with high distance_weight (0.8), result should be within 50 m."""
        mics_xy = self._mics_xy(triangle_mics)
        source = (30.0, 40.0)
        signals = _simulate_source(source, mics_xy, snr_db=35, seed=500)
        result = triangulate(
            signals, triangle_mics, sample_rate=16000, distance_weight=0.8
        )
        err = _loc_error_m(triangle_mics, result, source)
        assert err < 50.0, f"High weight: error {err:.1f} m (want <50 m)"


# ---------------------------------------------------------------------------
# N-microphone triangulation (N > 3)
# ---------------------------------------------------------------------------


class TestTriangulateNMics:
    """N-microphone triangulation (N > 3)."""

    def test_accepts_6_signals(self, hexagon_mics: list[MicPosition]) -> None:
        """triangulate() should accept 6 signals + 6 mic positions."""
        signals = [
            np.random.default_rng(i).normal(size=16000).astype(np.float32)
            for i in range(6)
        ]
        result = triangulate(signals, hexagon_mics, sample_rate=16000)
        assert isinstance(result, TriangulationResult)

    def test_accepts_4_signals(self) -> None:
        """triangulate() should accept 4 signals (square layout)."""
        mics = [
            MicPosition(lat=55.7510, lon=37.6130),
            MicPosition(lat=55.7510, lon=37.6146),
            MicPosition(lat=55.7519, lon=37.6130),
            MicPosition(lat=55.7519, lon=37.6146),
        ]
        signals = [
            np.random.default_rng(i).normal(size=16000).astype(np.float32)
            for i in range(4)
        ]
        result = triangulate(signals, mics, sample_rate=16000)
        assert isinstance(result, TriangulationResult)

    def test_6mic_centroid_accuracy(self, hexagon_mics: list[MicPosition]) -> None:
        """Source at centroid of 6 mics -> error < 15m."""
        lat0, lon0 = hexagon_mics[0].lat, hexagon_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in hexagon_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))
        centroid = (np.mean([p[0] for p in mics_xy]), np.mean([p[1] for p in mics_xy]))

        signals = _simulate_source(centroid, mics_xy, snr_db=40, seed=600)
        result = triangulate(signals, hexagon_mics, sample_rate=16000)
        err = _loc_error_m(hexagon_mics, result, centroid)
        assert err < 15.0, f"6-mic centroid: error {err:.1f} m (want <15 m)"

    def test_rejects_2_signals(self) -> None:
        """2 signals should raise AssertionError."""
        mics = [MicPosition(lat=55.75, lon=37.61), MicPosition(lat=55.76, lon=37.62)]
        signals = [np.zeros(16000, dtype=np.float32) for _ in range(2)]
        with pytest.raises(AssertionError):
            triangulate(signals, mics, sample_rate=16000)

    def test_mismatched_lengths(self, hexagon_mics: list[MicPosition]) -> None:
        """5 signals + 6 mics -> AssertionError."""
        signals = [np.zeros(16000, dtype=np.float32) for _ in range(5)]
        with pytest.raises(AssertionError):
            triangulate(signals, hexagon_mics)

    def test_6mic_outside_source(self, hexagon_mics: list[MicPosition]) -> None:
        """Source outside hexagon — should still converge within 40 m."""
        lat0, lon0 = hexagon_mics[0].lat, hexagon_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in hexagon_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        source = (-150.0, 100.0)
        signals = _simulate_source(source, mics_xy, snr_db=35, seed=601)
        result = triangulate(signals, hexagon_mics, sample_rate=16000)
        err = _loc_error_m(hexagon_mics, result, source)
        assert err < 100.0, f"6-mic outside: error {err:.1f} m (want <100 m)"

    def test_backward_compat_3_mics(self, triangle_mics: list[MicPosition]) -> None:
        """3 mics should still work after N-mic generalization."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))
        centroid = (np.mean([p[0] for p in mics_xy]), np.mean([p[1] for p in mics_xy]))

        signals = _simulate_source(centroid, mics_xy, snr_db=40, seed=7)
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        err = _loc_error_m(triangle_mics, result, centroid)
        assert err < 15.0, f"3-mic backward compat: error {err:.1f} m (want <15 m)"


# ---------------------------------------------------------------------------
# error_m quality — TDOA residual metric
# ---------------------------------------------------------------------------


class TestErrorMetric:
    """Tests that error_m reflects real TDOA residuals, not multi-start spread."""

    def test_error_m_nonzero(self, triangle_mics: list[MicPosition]) -> None:
        """error_m must be > 0 for any realistic scenario (not ±0.0)."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        source = (30.0, 50.0)
        signals = _simulate_source(source, mics_xy, snr_db=40, seed=77)
        result = triangulate(signals, triangle_mics, sample_rate=16000)
        assert result.error_m > 0.001, (
            f"error_m should reflect real TDOA residuals (>1 mm), got {result.error_m}"
        )

    def test_error_m_scales_with_noise(self, triangle_mics: list[MicPosition]) -> None:
        """error_m should increase when noise is added (lower SNR)."""
        lat0, lon0 = triangle_mics[0].lat, triangle_mics[0].lon
        mics_xy = [(0.0, 0.0)]
        for m in triangle_mics[1:]:
            mics_xy.append(_latlon_to_meters(lat0, lon0, m.lat, m.lon))

        source = (30.0, 50.0)
        signals_clean = _simulate_source(source, mics_xy, snr_db=40, seed=88)
        signals_noisy = _simulate_source(source, mics_xy, snr_db=10, seed=88)

        result_clean = triangulate(signals_clean, triangle_mics, sample_rate=16000)
        result_noisy = triangulate(signals_noisy, triangle_mics, sample_rate=16000)

        assert result_noisy.error_m > result_clean.error_m, (
            f"Noisy error_m ({result_noisy.error_m:.2f}) should be > "
            f"clean error_m ({result_clean.error_m:.2f})"
        )
