import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path

DEMO_AUDIO_DIR = Path(__file__).parent.parent.parent / "demo" / "audio"
SPEED_OF_SOUND = 343.0
SAMPLE_RATE = 16000

MIC_DISTANCES = {
    "A": 0,
    "B": 45,
    "C": 72,
}


class MicSimulator:
    def __init__(self, scenario: str = "chainsaw"):
        self.scenario = scenario

    async def get_signals(self) -> tuple[list[np.ndarray], list[str]]:
        """
        Returns:
            signals: list of 3 numpy arrays (with artificial delays)
            paths:   list of 3 temp WAV file paths
        """
        await asyncio.sleep(0.5)  # simulate capture time

        audio_file = DEMO_AUDIO_DIR / f"{self.scenario}.wav"
        if not audio_file.exists():
            audio_file = DEMO_AUDIO_DIR / "silence.wav"

        if audio_file.exists():
            waveform, sr = sf.read(str(audio_file), dtype="float32")
        else:
            # Last-resort fallback: generate silence in memory
            waveform = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
            sr = SAMPLE_RATE
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        signals = []
        paths = []

        for mic_id, distance_m in MIC_DISTANCES.items():
            delay_samples = int(distance_m / SPEED_OF_SOUND * SAMPLE_RATE)
            delayed = np.concatenate([np.zeros(delay_samples), waveform])

            # Add small noise
            noise = np.random.normal(0, 0.002, len(delayed)).astype(np.float32)
            delayed = delayed + noise

            signals.append(delayed)

            # Write to temp file for YAMNet
            import tempfile, os

            tmp = tempfile.NamedTemporaryFile(suffix=f"_mic_{mic_id}.wav", delete=False)
            sf.write(tmp.name, delayed, SAMPLE_RATE)
            paths.append(tmp.name)

        return signals, paths
