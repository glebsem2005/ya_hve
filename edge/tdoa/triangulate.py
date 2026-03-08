import numpy as np
from scipy.optimize import minimize
from scipy.fft import rfft, irfft
from scipy.signal import butter, sosfilt
from dataclasses import dataclass

from edge.tdoa.distance import estimate_distances

SPEED_OF_SOUND = 343.0  # m/s at ~20C


@dataclass
class MicPosition:
    lat: float
    lon: float


@dataclass
class TriangulationResult:
    lat: float
    lon: float
    error_m: float


def _latlon_to_meters(lat1, lon1, lat2, lon2) -> tuple[float, float]:
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    x = dlon * R * np.cos(np.radians(lat1))
    y = dlat * R
    return x, y


def _meters_to_latlon(lat0, lon0, dx, dy) -> tuple[float, float]:
    R = 6371000
    dlat = np.degrees(dy / R)
    dlon = np.degrees(dx / (R * np.cos(np.radians(lat0))))
    return lat0 + dlat, lon0 + dlon


def _bandpass_filter(
    signal: np.ndarray, sr: int, low: float = 200.0, high: float = 6000.0
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to remove low-freq rumble and high-freq noise.

    Chainsaw fundamental: 200-1000 Hz, gunshot transient: 500-5000 Hz.
    The 200-6000 Hz band covers all threat classes while rejecting wind/electronics noise.
    """
    nyq = sr / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return signal
    sos = butter(4, [low_n, high_n], btype="bandpass", output="sos")
    return sosfilt(sos, signal).astype(signal.dtype)


def _estimate_tdoa(sig_a: np.ndarray, sig_b: np.ndarray, sr: int) -> float:
    # TDOA-01: pad signals to equal length
    max_len = max(len(sig_a), len(sig_b))
    if len(sig_a) < max_len:
        sig_a = np.pad(sig_a, (0, max_len - len(sig_a)))
    if len(sig_b) < max_len:
        sig_b = np.pad(sig_b, (0, max_len - len(sig_b)))

    # GCC-PHAT with beta=0.75 (soft whitening)
    n = 2 * max_len - 1
    SIG_A = rfft(sig_a, n)
    SIG_B = rfft(sig_b, n)
    R = np.conj(SIG_A) * SIG_B
    denom = np.abs(R) ** 0.75 + 1e-10
    cc = np.real(irfft(R / denom, n))

    # Shift so that zero-lag is at the center (index max_len - 1)
    cc = np.roll(cc, max_len - 1)

    # No abs() — avoid promoting anti-correlation multipath peaks
    peak = int(np.argmax(cc))

    # Subpixel quadratic (parabolic) interpolation
    if 0 < peak < len(cc) - 1:
        y_m = float(cc[peak - 1])
        y_0 = float(cc[peak])
        y_p = float(cc[peak + 1])
        denom_q = y_m - 2 * y_0 + y_p
        if abs(denom_q) > 1e-10:
            peak = peak + 0.5 * (y_m - y_p) / denom_q

    lag = peak - (max_len - 1)
    return lag / sr


def _signal_snr(signal: np.ndarray) -> float:
    """Estimate SNR in dB. Assumes signal has both quiet and loud sections."""
    if len(signal) == 0:
        return 0.0
    frame_size = min(512, len(signal))
    n_frames = len(signal) // frame_size
    if n_frames < 2:
        return 0.0
    energies = np.array(
        [
            np.mean(signal[i * frame_size : (i + 1) * frame_size] ** 2)
            for i in range(n_frames)
        ]
    )
    noise_floor = np.percentile(energies, 10) + 1e-15
    signal_peak = np.percentile(energies, 90) + 1e-15
    return float(10 * np.log10(signal_peak / noise_floor))


def triangulate(
    signals: list[np.ndarray],
    mic_positions: list[MicPosition],
    sample_rate: int = 16000,
    use_bandpass: bool = True,
    temperature_c: float | None = None,
    distance_weight: float = 0.3,
    zone_type: str = "default",
) -> TriangulationResult:
    """Triangulate sound source using TDOA + energy-based distance fusion.

    Combines two complementary approaches:
    - TDOA (GCC-PHAT): constrains *difference* of distances between mic pairs
    - Energy-based ranging: constrains *absolute* distance to each mic

    Neither method is accurate alone, but together they narrow the
    uncertainty region significantly.

    Parameters
    ----------
    distance_weight : float
        Relative weight of distance constraints vs TDOA constraints
        in the cost function.  0.0 = pure TDOA, 1.0 = equal weight.
        Default 0.3 (TDOA dominates, distance adds soft constraint).
    """
    assert len(signals) == 3 and len(mic_positions) == 3

    mic_a, mic_b, mic_c = mic_positions
    sig_a, sig_b, sig_c = signals

    # Temperature-corrected speed of sound (if provided)
    speed = SPEED_OF_SOUND
    if temperature_c is not None:
        speed = 331.3 + 0.606 * temperature_c

    lat0, lon0 = mic_a.lat, mic_a.lon

    bx, by = _latlon_to_meters(lat0, lon0, mic_b.lat, mic_b.lon)
    cx, cy = _latlon_to_meters(lat0, lon0, mic_c.lat, mic_c.lon)

    mics_m = np.array([[0, 0], [bx, by], [cx, cy]])

    # Apply bandpass filter to improve cross-correlation quality
    if use_bandpass:
        sig_a = _bandpass_filter(sig_a, sample_rate)
        sig_b = _bandpass_filter(sig_b, sample_rate)
        sig_c = _bandpass_filter(sig_c, sample_rate)

    # Compute ALL 3 TDOA pairs for overdetermined system
    tdoa_ab = _estimate_tdoa(sig_a, sig_b, sample_rate)
    tdoa_ac = _estimate_tdoa(sig_a, sig_c, sample_rate)
    tdoa_bc = _estimate_tdoa(sig_b, sig_c, sample_rate)

    d_ab = tdoa_ab * speed
    d_ac = tdoa_ac * speed
    d_bc = tdoa_bc * speed

    # SNR-based weights: higher SNR pairs get more influence
    snr_a = _signal_snr(sig_a)
    snr_b = _signal_snr(sig_b)
    snr_c = _signal_snr(sig_c)
    # Weight for each pair = arithmetic mean of the two mics' SNR
    w_ab = max(1.0, (snr_a + snr_b) / 2.0)
    w_ac = max(1.0, (snr_a + snr_c) / 2.0)
    w_bc = max(1.0, (snr_b + snr_c) / 2.0)

    # Energy-based distance estimates (inverse-square law)
    dist_estimates = estimate_distances(
        [sig_a, sig_b, sig_c],
        [snr_a, snr_b, snr_c],
        zone_type=zone_type,
    )
    est_da = dist_estimates[0].distance_m
    est_db = dist_estimates[1].distance_m
    est_dc = dist_estimates[2].distance_m
    # Per-mic confidence weights for distance constraints
    conf_a = dist_estimates[0].confidence
    conf_b = dist_estimates[1].confidence
    conf_c = dist_estimates[2].confidence

    # Normalisation factors so that distance_weight actually controls the balance.
    # Without this, SNR-based TDOA weights (~40 dB) dwarf distance confidence (~0.4),
    # making distance_weight meaningless.
    w_tdoa_sum = w_ab + w_ac + w_bc + 1e-10
    w_dist_sum = conf_a + conf_b + conf_c + 1e-10

    def cost(pos):
        x, y = pos
        da = np.sqrt(x**2 + y**2)
        db = np.sqrt((x - bx) ** 2 + (y - by) ** 2)
        dc = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # TDOA constraints (difference of distances)
        # _estimate_tdoa(a, b) > 0 means B is farther, i.e. TDOA = (dist_B - dist_A) / speed
        err_ab = (db - da) - d_ab
        err_ac = (dc - da) - d_ac
        err_bc = (dc - db) - d_bc
        tdoa_cost = (
            w_ab * err_ab**2 + w_ac * err_ac**2 + w_bc * err_bc**2
        ) / w_tdoa_sum

        # Distance constraints (absolute distance to each mic)
        derr_a = (da - est_da) ** 2
        derr_b = (db - est_db) ** 2
        derr_c = (dc - est_dc) ** 2
        dist_cost = (conf_a * derr_a + conf_b * derr_b + conf_c * derr_c) / w_dist_sum

        return tdoa_cost + distance_weight * dist_cost

    # Multi-start optimization: try centroid + each mic position as initial guess
    centroid_x = mics_m[:, 0].mean()
    centroid_y = mics_m[:, 1].mean()

    initial_guesses = [
        [centroid_x, centroid_y],
        [0.0, 0.0],  # near mic A
        [bx, by],  # near mic B
        [cx, cy],  # near mic C
        [centroid_x * 2, centroid_y * 2],  # outside triangle
    ]

    best_result = None
    best_cost = float("inf")

    for x0 in initial_guesses:
        result = minimize(
            cost,
            x0,
            method="Nelder-Mead",
            options={"xatol": 0.1, "fatol": 0.01, "maxiter": 500},
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    src_x, src_y = best_result.x
    src_lat, src_lon = _meters_to_latlon(lat0, lon0, src_x, src_y)

    error_m = float(np.sqrt(best_result.fun)) if best_result.fun > 0 else 0.0

    return TriangulationResult(lat=src_lat, lon=src_lon, error_m=error_m)
