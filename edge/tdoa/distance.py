"""Energy-based distance estimation for audio signals.

Uses the inverse-square law: sound intensity falls as 1/r².
Given RMS energy of a signal at a microphone, we can estimate the
relative distance from the source to each mic.  With a known
reference level (calibration), absolute distances are available.

The estimates are noisy (wind, reflections, mic gain differences),
so they should be fused with TDOA constraints rather than used alone.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationProfile:
    """Zone-specific calibration for energy-based distance estimation.

    Different forest zone types have different acoustic propagation
    characteristics (canopy density, ground absorption, humidity).
    """

    zone_type: str  # "dense_forest", "clearing", "wetland", ...
    ref_rms: float
    ref_distance: float
    attenuation_factor: float


CALIBRATION_PROFILES: dict[str, CalibrationProfile] = {
    "default": CalibrationProfile("default", 0.25, 1.0, 1.0),
    "exploitation": CalibrationProfile("exploitation", 0.25, 1.0, 1.0),
    "oopt": CalibrationProfile("oopt", 0.22, 1.0, 1.3),
    "water_protection": CalibrationProfile("water_protection", 0.23, 1.0, 1.2),
    "protective_strip": CalibrationProfile("protective_strip", 0.24, 1.0, 1.1),
    "anti_erosion": CalibrationProfile("anti_erosion", 0.22, 1.0, 1.3),
    "spawning_protection": CalibrationProfile("spawning_protection", 0.20, 1.0, 1.4),
}


@dataclass
class DistanceEstimate:
    """Per-microphone distance estimate with uncertainty."""

    distance_m: float
    confidence: float  # 0..1, higher = more reliable


# Default reference: a chainsaw at 1 m produces ~0.25 RMS on int16-normalised float signal.
# This is a rough calibration constant; real deployments should measure it.
_DEFAULT_REF_RMS = 0.25
_DEFAULT_REF_DISTANCE = 1.0  # metres


def _rms_energy(signal: np.ndarray) -> float:
    """Root-mean-square energy of a signal."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def _energy_confidence(rms: float, snr_db: float) -> float:
    """Heuristic confidence in the energy-based distance estimate.

    High SNR and moderate RMS → high confidence.
    Very low RMS (far away / quiet) or low SNR → low confidence.
    """
    # SNR component: sigmoid-like mapping, 0 dB → ~0.1, 20 dB → ~0.8
    snr_conf = float(np.clip(snr_db / 25.0, 0.05, 1.0))
    # RMS component: very low energy means we're guessing
    rms_conf = float(np.clip(rms / 0.01, 0.1, 1.0))
    return min(snr_conf * rms_conf, 1.0)


def estimate_distances(
    signals: list[np.ndarray],
    snrs: list[float],
    ref_rms: float = _DEFAULT_REF_RMS,
    ref_distance: float = _DEFAULT_REF_DISTANCE,
    max_distance: float = 2000.0,
    zone_type: str = "default",
) -> list[DistanceEstimate]:
    """Estimate distance from sound source to each microphone.

    Uses the inverse-square law:  rms ∝ 1/r  (amplitude, not power).
    So  r = ref_distance * ref_rms / rms.

    Parameters
    ----------
    signals : list of numpy arrays (one per microphone)
    snrs : list of SNR values in dB (one per microphone)
    ref_rms : RMS amplitude of the reference source at ref_distance
    ref_distance : reference distance in metres
    max_distance : clamp estimates to this value (prevents ∞ for silence)
    zone_type : calibration profile name (see CALIBRATION_PROFILES)

    Returns
    -------
    list of DistanceEstimate, one per microphone
    """
    # Apply calibration profile if zone_type is recognized.
    # Explicit ref_rms / ref_distance kwargs still take priority when
    # they differ from the module-level defaults (backward compat).
    profile = CALIBRATION_PROFILES.get(zone_type)
    if profile is not None:
        if ref_rms == _DEFAULT_REF_RMS:
            ref_rms = profile.ref_rms
        if ref_distance == _DEFAULT_REF_DISTANCE:
            ref_distance = profile.ref_distance

    estimates = []
    for sig, snr in zip(signals, snrs):
        rms = _rms_energy(sig)
        if rms < 1e-10:
            # Essentially silence — can't estimate
            estimates.append(DistanceEstimate(distance_m=max_distance, confidence=0.0))
            continue

        dist = ref_distance * ref_rms / rms
        dist = float(np.clip(dist, 0.1, max_distance))
        conf = _energy_confidence(rms, snr)
        estimates.append(DistanceEstimate(distance_m=dist, confidence=conf))

    return estimates
