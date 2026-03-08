"""ForestGuard edge server — onset-triggered pipeline.

The microphone continuously monitors audio. ONLY when a sharp sound
transient is detected (energy spike) does the server proceed to:
  1. Classify the audio (YAMNet)
  2. Triangulate the source (GCC-PHAT TDOA)
  3. Send the result to the cloud via LoRa gateway

This replaces the old 30-second polling loop with event-driven detection.
"""

import asyncio
import logging
import os
import numpy as np
import soundfile as sf
import tempfile

from edge.audio.classifier import classify
from edge.audio.ndsi import compute_ndsi
from edge.audio.onset import OnsetDetector, FRAME_SIZE, LONG_TERM_FRAMES
from edge.tdoa.triangulate import triangulate, MicPosition, TriangulationResult
from edge.decision.decider import decide
from edge.drone.simulated import SimulatedDrone
from simulator.lora.socket_relay import LoraRelay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_mic_positions() -> list[MicPosition]:
    """Load microphone positions from database, fallback to env vars."""
    try:
        from cloud.db.microphones import get_online

        online_mics = get_online()
        if len(online_mics) >= 3:
            # Use first 3 online mics for TDOA triangulation
            return [MicPosition(lat=m.lat, lon=m.lon) for m in online_mics[:3]]
        logger.warning(
            "Only %d online mics in DB, falling back to env vars", len(online_mics)
        )
    except Exception as e:
        logger.warning("Failed to load mics from DB: %s, using env vars", e)

    return [
        MicPosition(
            lat=float(os.getenv("MIC_A_LAT", 55.7510)),
            lon=float(os.getenv("MIC_A_LON", 37.6130)),
        ),
        MicPosition(
            lat=float(os.getenv("MIC_B_LAT", 55.7515)),
            lon=float(os.getenv("MIC_B_LON", 37.6138)),
        ),
        MicPosition(
            lat=float(os.getenv("MIC_C_LAT", 55.7508)),
            lon=float(os.getenv("MIC_C_LON", 37.6142)),
        ),
    ]


MIC_POSITIONS = _load_mic_positions()

SAMPLE_RATE = 16000
LISTEN_CHUNK_SECONDS = 2  # continuous listening window
RECORD_SECONDS = 3  # full recording after onset trigger


async def _acquire_signals_sim(scenario: str):
    """Acquire signals from simulator (demo mode)."""
    from simulator.audio.mic_stream import MicSimulator

    mic_sim = MicSimulator(scenario)
    return await mic_sim.get_signals()


async def _acquire_signals_real(device_id: int):
    """Acquire signals from single real microphone."""
    from simulator.audio.real_mic import RealMic

    mic = RealMic(device_id=device_id)
    return await mic.get_signals()


async def _acquire_signals_real3(device_ids: list[int]):
    """Acquire signals from 3-mic real array."""
    from simulator.audio.real_mic import RealMicArray

    mic = RealMicArray(device_ids=device_ids)
    return await mic.get_signals()


async def _continuous_listen_real(device_id: int, detector: OnsetDetector):
    """Continuously record short chunks and check for onset.

    Returns signals + paths only when a sharp sound is detected.
    Otherwise keeps listening indefinitely.
    """
    import sounddevice as sd

    logger.info("Listening on device %d for sharp sounds...", device_id)
    loop = asyncio.get_event_loop()

    while True:
        # Record a short chunk
        recording = await loop.run_in_executor(
            None,
            lambda: sd.rec(
                int(SAMPLE_RATE * LISTEN_CHUNK_SECONDS),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_id,
                blocking=True,
            ),
        )
        waveform = recording[:, 0]

        # Check for onset
        event = detector.detect(waveform, SAMPLE_RATE)
        if event.triggered:
            logger.info(
                "Sharp sound detected! ratio=%.1f energy=%.4f",
                event.energy_ratio,
                event.peak_energy,
            )
            # Record extended chunk for classification
            extended = await loop.run_in_executor(
                None,
                lambda: sd.rec(
                    int(SAMPLE_RATE * RECORD_SECONDS),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=device_id,
                    blocking=True,
                ),
            )
            full_waveform = np.concatenate([waveform, extended[:, 0]])

            # Write to temp files + create fake TDOA delays for single-mic mode
            tmp = tempfile.NamedTemporaryFile(suffix="_onset.wav", delete=False)
            sf.write(tmp.name, full_waveform, SAMPLE_RATE)

            sig_a = full_waveform
            sig_b = np.concatenate([np.zeros(5, dtype=np.float32), full_waveform])
            sig_c = np.concatenate([np.zeros(15, dtype=np.float32), full_waveform])

            return [sig_a, sig_b, sig_c], [tmp.name, tmp.name, tmp.name]

        # Small pause to prevent CPU spin
        await asyncio.sleep(0.05)


async def _continuous_listen_real3(device_ids: list[int], detector: OnsetDetector):
    """Continuously listen on 3-mic array, trigger on onset."""
    import sounddevice as sd

    logger.info("Listening on devices %s for sharp sounds...", device_ids)
    loop = asyncio.get_event_loop()
    primary_device = device_ids[0]

    while True:
        # Monitor primary mic for onset
        recording = await loop.run_in_executor(
            None,
            lambda: sd.rec(
                int(SAMPLE_RATE * LISTEN_CHUNK_SECONDS),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=primary_device,
                blocking=True,
            ),
        )
        waveform = recording[:, 0]

        event = detector.detect(waveform, SAMPLE_RATE)
        if event.triggered:
            logger.info(
                "Sharp sound detected! ratio=%.1f — recording all 3 mics",
                event.energy_ratio,
            )
            # Now record from all 3 mics simultaneously
            from simulator.audio.real_mic import RealMicArray

            mic = RealMicArray(device_ids=device_ids)
            return await mic.get_signals()

        await asyncio.sleep(0.05)


async def _continuous_listen_sim(scenario: str, detector: OnsetDetector):
    """Simulated onset detection for demo mode.

    In sim mode we load the demo audio and run onset detection on it.
    If the scenario is a threat (chainsaw/gunshot), onset will trigger.
    If it's normal (birds), onset may not trigger — which is correct behavior.
    """
    from simulator.audio.mic_stream import MicSimulator

    logger.info("Simulated listening for scenario: %s", scenario)

    mic_sim = MicSimulator(scenario)
    signals, audio_paths = await mic_sim.get_signals()

    # Run onset detection on the first signal
    event = detector.detect(signals[0], SAMPLE_RATE)

    if event.triggered:
        logger.info(
            "Onset triggered (sim): ratio=%.1f energy=%.4f",
            event.energy_ratio,
            event.peak_energy,
        )
        return signals, audio_paths
    else:
        logger.info(
            "No sharp sound in scenario '%s' (ratio=%.1f < threshold) — skipping",
            scenario,
            event.energy_ratio,
        )
        return None, None


async def run():
    lora = LoraRelay(
        host=os.getenv("LORA_GATEWAY_HOST", "lora_gateway"),
        port=int(os.getenv("LORA_GATEWAY_PORT", 9000)),
    )
    drone = SimulatedDrone(
        home_lat=MIC_POSITIONS[0].lat,
        home_lon=MIC_POSITIONS[0].lon,
    )

    detector = OnsetDetector()

    logger.info("Edge server started — onset-triggered mode")
    logger.info("Microphone will trigger ONLY on sharp sounds")

    sim_already_fired = False

    try:
        while True:
            try:
                mic_mode = os.getenv("MIC_MODE", "sim")
                signals = None
                audio_paths = None

                # In sim mode, don't re-run the same scenario in a
                # tight loop.  Wait for the next demo API call instead.
                if mic_mode == "sim" and sim_already_fired:
                    await asyncio.sleep(30)
                    continue

                # --- Onset-triggered acquisition ---
                try:
                    if mic_mode == "real":
                        device_id = int(os.getenv("MIC_DEVICE_ID", 0))
                        signals, audio_paths = await _continuous_listen_real(
                            device_id, detector
                        )

                    elif mic_mode == "real_3":
                        ids = [
                            int(x)
                            for x in os.getenv("MIC_DEVICE_IDS", "0,1,2").split(",")
                        ]
                        signals, audio_paths = await _continuous_listen_real3(
                            ids, detector
                        )

                    else:
                        scenario = os.getenv("DEMO_SCENARIO", "chainsaw")
                        signals, audio_paths = await _continuous_listen_sim(
                            scenario, detector
                        )

                        if signals is None:
                            # No onset in sim mode — wait and retry
                            await asyncio.sleep(10)
                            detector.reset()
                            continue

                        # In sim mode, process once then wait for next
                        # demo trigger (don't re-trigger the same scenario
                        # in a tight loop — that causes constant pings)
                        sim_already_fired = True

                except Exception as e:
                    logger.error("Failed to acquire audio signals: %s", e)
                    await asyncio.sleep(5)
                    continue

                # --- Classify (only runs after onset trigger) ---
                try:
                    audio = classify(audio_paths[0])
                except Exception as e:
                    logger.error("Classification failed: %s", e)
                    await asyncio.sleep(5)
                    continue

                logger.info(
                    "Classification: %s (%.0f%%)", audio.label, audio.confidence * 100
                )

                # --- NDSI analysis ---
                ndsi_result = compute_ndsi(signals[0], SAMPLE_RATE)
                logger.info(
                    "NDSI: %.3f (%s)", ndsi_result.ndsi, ndsi_result.interpretation
                )

                # --- Triangulate ---
                try:
                    location = triangulate(signals, MIC_POSITIONS)
                except Exception as e:
                    centroid_lat = sum(m.lat for m in MIC_POSITIONS) / len(
                        MIC_POSITIONS
                    )
                    centroid_lon = sum(m.lon for m in MIC_POSITIONS) / len(
                        MIC_POSITIONS
                    )
                    location = TriangulationResult(
                        lat=centroid_lat, lon=centroid_lon, error_m=999.0
                    )
                    logger.warning(
                        "Triangulation failed, falling back to centroid: %s", e
                    )

                logger.info(
                    "Source location: %.4f°N %.4f°E (±%.0fm)",
                    location.lat,
                    location.lon,
                    location.error_m,
                )

                decision = decide(audio, location, ndsi=ndsi_result)
                logger.info("Decision: %s", decision.reason)

                if not decision.send_drone and not decision.send_lora:
                    logger.info("No action needed — skipping drone and notification")
                    await asyncio.sleep(5)
                    detector.reset()
                    continue

                if decision.send_drone:
                    # --- Drone operations ---
                    photo = None
                    try:
                        await asyncio.wait_for(drone.takeoff(), timeout=10)
                        async for pos in drone.fly_to(location.lat, location.lon):
                            logger.info("   Drone at %.4f°N %.4f°E", pos.lat, pos.lon)
                        photo = await asyncio.wait_for(
                            drone.capture_photo(), timeout=10
                        )
                        await asyncio.wait_for(drone.return_home(), timeout=10)
                    except asyncio.TimeoutError:
                        logger.error("Drone operation timed out")
                        photo = None
                    except Exception as e:
                        logger.error("Drone operation failed: %s", e)
                        photo = None

                    # --- LoRa send to gateway (→ Yandex GPT → Telegram + dashboard) ---
                    packet = {
                        "class": audio.label,
                        "confidence": audio.confidence,
                        "lat": location.lat,
                        "lon": location.lon,
                        "priority": decision.priority,
                        "error_m": location.error_m,
                        "photo_b64": photo.b64 if photo else None,
                        "ndsi": ndsi_result.ndsi,
                        "ndsi_interpretation": ndsi_result.interpretation,
                    }
                    try:
                        await asyncio.wait_for(lora.send(packet), timeout=10)
                        logger.info("LoRa packet sent to gateway")
                    except asyncio.TimeoutError:
                        logger.error("LoRa send timed out — packet lost")
                    except Exception as e:
                        logger.error("LoRa send failed — packet lost: %s", e)

                # Cooldown before next detection cycle
                await asyncio.sleep(5)
                detector.reset()

            except Exception as e:
                logger.critical("Unexpected error in main loop: %s", e)
                await asyncio.sleep(10)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return


if __name__ == "__main__":
    asyncio.run(run())
