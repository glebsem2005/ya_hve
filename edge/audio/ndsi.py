"""NDSI (Normalized Difference Soundscape Index) tracker.

NDSI = (B - A) / (B + A)
A = anthropogenic band energy (1-2 kHz)
B = biophonic band energy (2-11 kHz)

Reference values from NB02 v5:
  chainsaw = -0.653, gunshot = -0.794, engine = -0.689
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NDSIResult:
    """Result of NDSI computation for an audio waveform."""

    ndsi: float  # -1.0 to 1.0
    anthro_energy: float  # energy in 1-2 kHz
    bio_energy: float  # energy in 2-11 kHz
    interpretation: str  # human-readable


def compute_ndsi(waveform: np.ndarray, sr: int = 16000) -> NDSIResult:
    """Compute NDSI for audio waveform.

    Uses FFT to compute energy in anthropogenic (1-2 kHz) and
    biophonic (2-11 kHz) bands.

    Parameters
    ----------
    waveform : numpy array of audio samples
    sr : sample rate in Hz

    Returns
    -------
    NDSIResult with NDSI value, band energies, and interpretation
    """
    if len(waveform) == 0:
        return NDSIResult(
            ndsi=0.0,
            anthro_energy=0.0,
            bio_energy=0.0,
            interpretation="empty signal",
        )

    # FFT
    spectrum = np.abs(np.fft.rfft(waveform.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sr)

    # Band masks
    anthro_mask = (freqs >= 1000) & (freqs <= 2000)
    bio_mask = (freqs >= 2000) & (freqs <= min(11000, sr / 2))

    anthro_energy = float(np.sum(spectrum[anthro_mask] ** 2))
    bio_energy = float(np.sum(spectrum[bio_mask] ** 2))

    denom = bio_energy + anthro_energy
    if denom < 1e-15:
        return NDSIResult(
            ndsi=0.0,
            anthro_energy=0.0,
            bio_energy=0.0,
            interpretation="silence",
        )

    ndsi = (bio_energy - anthro_energy) / denom

    if ndsi < -0.5:
        interpretation = "strong anthropogenic activity"
    elif ndsi < 0.0:
        interpretation = "moderate anthropogenic activity"
    elif ndsi < 0.5:
        interpretation = "mixed soundscape"
    else:
        interpretation = "natural soundscape"

    return NDSIResult(
        ndsi=round(ndsi, 4),
        anthro_energy=round(anthro_energy, 6),
        bio_energy=round(bio_energy, 6),
        interpretation=interpretation,
    )
