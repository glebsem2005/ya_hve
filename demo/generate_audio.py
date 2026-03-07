"""Generate synthetic demo WAV files for edge simulator.

Creates minimal WAV files so MicSimulator doesn't crash when
demo/audio/ is empty. Idempotent — skips files that already exist.

Usage:
    python demo/generate_audio.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "audio"
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION


def _save(name: str, waveform: np.ndarray) -> None:
    path = AUDIO_DIR / name
    if path.exists():
        return
    sf.write(str(path), waveform.astype(np.float32), SAMPLE_RATE)
    print(f"  generated {name} ({path.stat().st_size // 1024} KB)")


def generate_silence():
    _save("silence.wav", np.zeros(N_SAMPLES, dtype=np.float32))


def generate_chainsaw():
    """Sawtooth ~100-400 Hz + lowpass noise — crude chainsaw approximation."""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    # Frequency sweep 100-400 Hz
    freq = 100 + 300 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    phase = np.cumsum(freq / SAMPLE_RATE)
    saw = 2.0 * (phase % 1.0) - 1.0
    # Add noise
    noise = np.random.default_rng(42).normal(0, 0.3, N_SAMPLES)
    mixed = 0.6 * saw + 0.4 * noise
    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    _save("chainsaw.wav", mixed)


def generate_gunshot():
    """Short impulse with exponential decay."""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    impulse = np.zeros(N_SAMPLES, dtype=np.float32)
    # Place impulse at 0.5s
    start = int(0.5 * SAMPLE_RATE)
    burst_len = int(0.01 * SAMPLE_RATE)  # 10ms burst
    rng = np.random.default_rng(7)
    impulse[start : start + burst_len] = rng.uniform(-1.0, 1.0, burst_len)
    # Exponential decay over 0.3s
    decay_len = int(0.3 * SAMPLE_RATE)
    decay_env = np.exp(-np.linspace(0, 8, decay_len))
    impulse[start : start + decay_len] *= np.pad(
        decay_env, (0, max(0, decay_len - len(decay_env)))
    )[:decay_len]
    _save("gunshot.wav", np.clip(impulse, -1.0, 1.0))


def generate_normal():
    """Low-amplitude pink-ish noise — ambient forest."""
    rng = np.random.default_rng(0)
    white = rng.normal(0, 0.05, N_SAMPLES).astype(np.float32)
    _save("normal.wav", white)


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating demo audio files...")
    generate_silence()
    generate_chainsaw()
    generate_gunshot()
    generate_normal()
    print("Done.")


if __name__ == "__main__":
    main()
