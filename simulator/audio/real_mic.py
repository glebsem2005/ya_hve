import asyncio
import numpy as np
import soundfile as sf
import tempfile
import os
from dataclasses import dataclass

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 2
CHANNELS      = 1

@dataclass
class MicCapture:
    signals:     list
    audio_paths: list

class RealMic:

    def __init__(self, device_id: int | None = None):
        self.device_id = device_id

    async def get_signals(self) -> tuple[list, list]:
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        recording = await loop.run_in_executor(
            None,
            lambda: sd.rec(
                int(SAMPLE_RATE * CHUNK_SECONDS),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                device=self.device_id,
                blocking=True,
            )
        )

        waveform = recording[:, 0]  # mono

        tmp = tempfile.NamedTemporaryFile(suffix="_real_mic.wav", delete=False)
        sf.write(tmp.name, waveform, SAMPLE_RATE)

        # Fake small delays so TDOA doesn't crash
        delay_a = np.zeros(5, dtype=np.float32)
        delay_b = np.zeros(15, dtype=np.float32)

        sig_b = np.concatenate([delay_a, waveform])
        sig_c = np.concatenate([delay_b, waveform])

        return [sig_a, sig_b, sig_c], [tmp.name, tmp.name, tmp.name]


class RealMicArray:

    def __init__(self, device_ids: list[int]):
        assert len(device_ids) == 3, "Need exactly 3 device IDs"
        self.device_ids = device_ids

    async def get_signals(self) -> tuple[list, list]:
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        signals = []
        paths = []
        tasks = [
            loop.run_in_executor(
                None,
                lambda did=did: sd.rec(
                    int(SAMPLE_RATE * CHUNK_SECONDS),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    device=did,
                    blocking=True,
                )
            )
            for did in self.device_ids
        ]

        recordings = await asyncio.gather(*tasks)

        for i, rec in enumerate(recordings):
            waveform = rec[:, 0]
            tmp = tempfile.NamedTemporaryFile(
                suffix=f"_mic_{['A','B','C'][i]}.wav", delete=False
            )
            sf.write(tmp.name, waveform, SAMPLE_RATE)
            signals.append(waveform)
            paths.append(tmp.name)

        return signals, paths


def list_devices():
    import sounddevice as sd
    devices = sd.query_devices()
    print("\nАудио устройства:")
    print("─" * 50)
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"  [{i}] {d['name']}")
            print(f"       channels={d['max_input_channels']} sr={int(d['default_samplerate'])}Hz")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List audio devices")
    parser.add_argument("--device", type=int, default=None, help="Device ID")
    parser.add_argument("--seconds", type=int, default=5, help="Record duration")
    args = parser.parse_args()

    if args.list:
        list_devices()
    else:
        import sounddevice as sd
        from edge.audio.classifier import classify

        print(f"Recording {args.seconds}s from device {args.device or 'default'}...")
        print("Make some noise! (chainsaw sound, gunshot, birds...)\n")

        recording = sd.rec(
            int(SAMPLE_RATE * args.seconds),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            device=args.device,
            blocking=True,
        )

        tmp = tempfile.NamedTemporaryFile(suffix="_test.wav", delete=False)
        sf.write(tmp.name, recording[:, 0], SAMPLE_RATE)
        print(f"   Saved to {tmp.name}")

        result = classify(tmp.name)
        print(f"\nКлассификация: {result.label}")
        print(f"Уверенность:    {result.confidence:.0%}")
        print(f"Top scores:     {result.raw_scores}")
        os.unlink(tmp.name)
