"""Download real demo WAV files for edge simulator.

Downloads real audio samples from ESC-50 (GitHub) and UrbanSound8K
(HuggingFace) so base YAMNet classifies them correctly without
hardcoded overrides. Falls back to synthetic generation on error.

Output: demo/audio/{scenario}.wav — WAV, 16kHz, mono, 5s, peak-normalized.

Usage:
    python demo/generate_audio.py          # download (skip existing)
    python demo/generate_audio.py --force  # re-download all
"""

import io
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

AUDIO_DIR = Path(__file__).parent / "audio"
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION

# ESC-50 GitHub: multiple candidates per class (fallback chain)
ESC50_BASE = "https://github.com/karolpiczak/ESC-50/raw/master/audio"
ESC50_FILES: dict[str, list[str]] = {
    "chainsaw": ["1-116765-A-41.wav", "1-19898-A-41.wav", "1-19898-B-41.wav"],
    "engine": ["1-18527-A-44.wav", "1-18527-B-44.wav", "1-22882-A-44.wav"],
    "fire": ["1-17150-A-12.wav", "1-17565-A-12.wav", "1-17742-A-12.wav"],
    "axe": ["1-101336-A-30.wav", "1-103995-A-30.wav", "1-103999-A-30.wav"],
}


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------


def _process(data: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono, pad/trim to 5 s, peak-normalize."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        data = librosa.resample(
            data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
        )
    if len(data) > N_SAMPLES:
        data = data[:N_SAMPLES]
    elif len(data) < N_SAMPLES:
        data = np.pad(data, (0, N_SAMPLES - len(data)))
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak * 0.95
    return data.astype(np.float32)


def _save(name: str, waveform: np.ndarray) -> None:
    path = AUDIO_DIR / name
    sf.write(str(path), waveform, SAMPLE_RATE)
    print(f"  saved {name} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_esc50(scenario: str) -> np.ndarray | None:
    """Try ESC-50 candidates in order; return processed waveform or None."""
    for fname in ESC50_FILES.get(scenario, []):
        url = f"{ESC50_BASE}/{fname}"
        try:
            r = requests.get(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            data, sr = sf.read(io.BytesIO(r.content))
            print(f"  downloaded {scenario} from ESC-50 ({fname})")
            return _process(data, sr)
        except Exception as e:
            print(f"  ESC-50 {fname} failed: {e}")
    return None


def _download_gunshot_us8k() -> np.ndarray | None:
    """Extract a gun_shot sample from UrbanSound8K (HuggingFace parquet).

    Uses shard 02 which has the best gunshot samples (4s, high peak).
    Target file: 135528-6-2-0.wav — reliably classified by YAMNet v7.
    """
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        path = hf_hub_download(
            "danavery/urbansound8K",
            "data/train-00002-of-00016-887e0748205b6fa9.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(path)
        df = table.to_pandas()
        guns = df[df["classID"] == 6]
        # Prefer specific file known to classify well; fallback to first
        target = guns[guns["slice_file_name"] == "135528-6-2-0.wav"]
        gun = target.iloc[0] if len(target) > 0 else guns.iloc[0]
        data, sr = sf.read(io.BytesIO(gun["audio"]["bytes"]))
        print(f"  downloaded gunshot from UrbanSound8K ({gun['slice_file_name']})")
        return _process(data, sr)
    except Exception as e:
        print(f"  UrbanSound8K download failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Synthetic fallbacks (used only when download fails)
# ---------------------------------------------------------------------------


def _synth_chainsaw() -> np.ndarray:
    """Sawtooth 100-400 Hz + noise — crude chainsaw."""
    rng = np.random.default_rng(42)
    quiet_len = int(1.0 * SAMPLE_RATE)
    fade_len = int(0.05 * SAMPLE_RATE)
    saw_len = N_SAMPLES - quiet_len
    quiet = rng.normal(0, 0.005, quiet_len).astype(np.float32)
    t = np.linspace(0, saw_len / SAMPLE_RATE, saw_len, endpoint=False)
    freq = 100 + 300 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    phase = np.cumsum(freq / SAMPLE_RATE)
    saw = 2.0 * (phase % 1.0) - 1.0
    noise = rng.normal(0, 0.3, saw_len)
    chainsaw = (0.6 * saw + 0.4 * noise).astype(np.float32)
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    chainsaw[:fade_len] *= fade
    mixed = np.concatenate([quiet, chainsaw])
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _synth_gunshot() -> np.ndarray:
    """Short impulse with exponential decay."""
    impulse = np.zeros(N_SAMPLES, dtype=np.float32)
    start = int(0.5 * SAMPLE_RATE)
    burst_len = int(0.01 * SAMPLE_RATE)
    rng = np.random.default_rng(7)
    impulse[start : start + burst_len] = rng.uniform(-1.0, 1.0, burst_len)
    decay_len = int(0.3 * SAMPLE_RATE)
    decay_env = np.exp(-np.linspace(0, 8, decay_len))
    impulse[start : start + decay_len] *= np.pad(
        decay_env, (0, max(0, decay_len - len(decay_env)))
    )[:decay_len]
    return np.clip(impulse, -1.0, 1.0).astype(np.float32)


def _synth_engine() -> np.ndarray:
    """Low-frequency hum 60-120 Hz + noise."""
    rng = np.random.default_rng(99)
    quiet_len = int(1.0 * SAMPLE_RATE)
    fade_len = int(0.05 * SAMPLE_RATE)
    engine_len = N_SAMPLES - quiet_len
    quiet = rng.normal(0, 0.005, quiet_len).astype(np.float32)
    t = np.linspace(0, engine_len / SAMPLE_RATE, engine_len, endpoint=False)
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
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _synth_fire() -> np.ndarray:
    """Crackling noise — fire approximation."""
    rng = np.random.default_rng(55)
    noise = rng.normal(0, 0.3, N_SAMPLES).astype(np.float32)
    # Random amplitude modulation for crackle effect
    crackle = rng.exponential(0.3, N_SAMPLES).astype(np.float32)
    crackle = np.clip(crackle, 0, 1)
    return np.clip(noise * crackle, -1.0, 1.0).astype(np.float32)


def _synth_axe() -> np.ndarray:
    """Rhythmic impacts — axe chops on wood."""
    rng = np.random.default_rng(33)
    signal = rng.normal(0, 0.005, N_SAMPLES).astype(np.float32)
    # 4 chops at 0.8s intervals starting at 0.5s
    for i in range(4):
        start = int((0.5 + i * 0.8) * SAMPLE_RATE)
        if start >= N_SAMPLES:
            break
        burst_len = int(0.02 * SAMPLE_RATE)
        end = min(start + burst_len, N_SAMPLES)
        signal[start:end] = rng.uniform(-0.8, 0.8, end - start)
        # Decay
        decay_len = int(0.15 * SAMPLE_RATE)
        d_end = min(start + decay_len, N_SAMPLES)
        env = np.exp(-np.linspace(0, 6, d_end - start))
        signal[start:d_end] *= env[: d_end - start]
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


SYNTH_FALLBACKS: dict[str, callable] = {
    "chainsaw": _synth_chainsaw,
    "gunshot": _synth_gunshot,
    "engine": _synth_engine,
    "fire": _synth_fire,
    "axe": _synth_axe,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    force = "--force" in sys.argv
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading real demo audio files...")

    scenarios = ["chainsaw", "gunshot", "engine", "fire", "axe"]

    for scenario in scenarios:
        path = AUDIO_DIR / f"{scenario}.wav"
        if path.exists() and not force:
            print(f"  {scenario}.wav exists, skipping (use --force to re-download)")
            continue

        # Try real download first
        waveform = None
        if scenario == "gunshot":
            waveform = _download_gunshot_us8k()
        elif scenario in ESC50_FILES:
            waveform = _download_esc50(scenario)

        # Fallback to synthetic
        if waveform is None:
            fallback = SYNTH_FALLBACKS.get(scenario)
            if fallback:
                waveform = fallback()
                print(f"  {scenario}: using synthetic fallback")

        if waveform is not None:
            _save(f"{scenario}.wav", waveform)

    # Normal + silence (always synthetic)
    for name, data in [
        (
            "normal.wav",
            np.random.default_rng(0).normal(0, 0.05, N_SAMPLES).astype(np.float32),
        ),
        ("silence.wav", np.zeros(N_SAMPLES, dtype=np.float32)),
    ]:
        path = AUDIO_DIR / name
        if path.exists() and not force:
            print(f"  {name} exists, skipping")
            continue
        _save(name, data)

    print("Done.")


if __name__ == "__main__":
    main()
