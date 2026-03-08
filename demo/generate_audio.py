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
    """Sawtooth ~100-400 Hz + lowpass noise — crude chainsaw approximation.

    Structure: 1.0s quiet forest ambience → 50ms fade-in → 4.0s chainsaw.
    The quiet lead-in lets the onset detector build a low baseline,
    so the chainsaw start produces a high energy ratio (>>8.0 threshold).
    """
    rng = np.random.default_rng(42)
    quiet_len = int(1.0 * SAMPLE_RATE)  # 1s quiet
    fade_len = int(0.05 * SAMPLE_RATE)  # 50ms fade-in
    saw_len = N_SAMPLES - quiet_len  # 4s chainsaw

    # Quiet forest ambience (very low amplitude noise)
    quiet = rng.normal(0, 0.005, quiet_len).astype(np.float32)

    # Chainsaw phase: sawtooth ~100-400 Hz + noise
    t = np.linspace(0, saw_len / SAMPLE_RATE, saw_len, endpoint=False)
    freq = 100 + 300 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    phase = np.cumsum(freq / SAMPLE_RATE)
    saw = 2.0 * (phase % 1.0) - 1.0
    noise = rng.normal(0, 0.3, saw_len)
    chainsaw = (0.6 * saw + 0.4 * noise).astype(np.float32)

    # Fade-in: linear ramp over first 50ms of chainsaw
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    chainsaw[:fade_len] *= fade

    mixed = np.concatenate([quiet, chainsaw])
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


def generate_engine():
    """Low-frequency hum 60-120 Hz with noise — vehicle/machinery engine.

    Structure: 1.0s quiet forest ambience -> 50ms fade-in -> 4.0s engine drone.
    """
    rng = np.random.default_rng(99)
    quiet_len = int(1.0 * SAMPLE_RATE)
    fade_len = int(0.05 * SAMPLE_RATE)
    engine_len = N_SAMPLES - quiet_len

    quiet = rng.normal(0, 0.005, quiet_len).astype(np.float32)

    t = np.linspace(0, engine_len / SAMPLE_RATE, engine_len, endpoint=False)
    # Fundamental 80 Hz + harmonics at 120 Hz, slight vibrato
    freq1 = 80 + 5 * np.sin(2 * np.pi * 0.3 * t)
    freq2 = 120 + 8 * np.sin(2 * np.pi * 0.5 * t)
    phase1 = np.cumsum(freq1 / SAMPLE_RATE)
    phase2 = np.cumsum(freq2 / SAMPLE_RATE)
    tone1 = np.sin(2 * np.pi * phase1)
    tone2 = 0.5 * np.sin(2 * np.pi * phase2)
    noise = rng.normal(0, 0.15, engine_len)
    engine = (0.5 * tone1 + 0.3 * tone2 + 0.2 * noise).astype(np.float32)

    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    engine[:fade_len] *= fade

    mixed = np.concatenate([quiet, engine])
    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    _save("engine.wav", mixed)


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
    generate_engine()
    generate_normal()
    print("Done.")


if __name__ == "__main__":
    main()
