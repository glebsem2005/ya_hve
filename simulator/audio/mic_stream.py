import asyncio
import math
import numpy as np
import soundfile as sf
from pathlib import Path

DEMO_AUDIO_DIR = Path(__file__).parent.parent.parent / "demo" / "audio"
SPEED_OF_SOUND = 343.0
SAMPLE_RATE = 16000

# Legacy fallback distances (meters from mic A)
MIC_DISTANCES = {
    "A": 0,
    "B": 45,
    "C": 72,
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MicSimulator:
    def __init__(
        self,
        scenario: str = "chainsaw",
        source_lat: float | None = None,
        source_lon: float | None = None,
        mic_positions: list[tuple[float, float]] | None = None,
    ):
        self.scenario = scenario
        self.source_lat = source_lat
        self.source_lon = source_lon
        self.mic_positions = mic_positions  # list of (lat, lon)

    async def get_signals(self) -> tuple[list[np.ndarray], list[str]]:
        """
        Returns:
            signals: list of numpy arrays (with realistic delays)
            paths:   list of temp WAV file paths
        """
        await asyncio.sleep(0.5)  # simulate capture time

        audio_file = DEMO_AUDIO_DIR / f"{self.scenario}.wav"
        if not audio_file.exists():
            audio_file = DEMO_AUDIO_DIR / "silence.wav"

        if audio_file.exists():
            waveform, sr = sf.read(str(audio_file), dtype="float32")
        else:
            waveform = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
            sr = SAMPLE_RATE
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Compute distances: geo-based if available, else legacy fixed
        if self.mic_positions and self.source_lat is not None:
            distances = {
                chr(65 + i): _haversine(self.source_lat, self.source_lon, lat, lon)
                for i, (lat, lon) in enumerate(self.mic_positions)
            }
        else:
            distances = MIC_DISTANCES

        min_dist = max(min(distances.values()), 1.0)  # avoid div-by-zero

        signals = []
        paths = []

        for mic_id, distance_m in distances.items():
            relative_dist = distance_m - min_dist
            delay_samples = int(relative_dist / SPEED_OF_SOUND * SAMPLE_RATE)
            delayed = np.concatenate([np.zeros(delay_samples), waveform])

            # Inverse-distance attenuation
            if distance_m > 0:
                attenuation = min_dist / distance_m
                delayed = delayed * attenuation

            # Add small noise
            noise = np.random.normal(0, 0.002, len(delayed)).astype(np.float32)
            delayed = (delayed + noise).astype(np.float32)

            signals.append(delayed)

            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=f"_mic_{mic_id}.wav", delete=False)
            sf.write(tmp.name, delayed, SAMPLE_RATE)
            paths.append(tmp.name)

        return signals, paths
