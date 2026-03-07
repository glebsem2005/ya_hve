import asyncio
import logging
import os
from edge.audio.classifier import classify
from edge.tdoa.triangulate import triangulate, MicPosition, TriangulationResult
from edge.decision.decider import decide
from edge.drone.simulated import SimulatedDrone
from simulator.lora.socket_relay import LoraRelay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIC_POSITIONS = [
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


async def run():
    lora = LoraRelay(
        host=os.getenv("LORA_GATEWAY_HOST", "lora_gateway"),
        port=int(os.getenv("LORA_GATEWAY_PORT", 9000)),
    )
    drone = SimulatedDrone(
        home_lat=MIC_POSITIONS[0].lat,
        home_lon=MIC_POSITIONS[0].lon,
    )

    logger.info("Edge server started — listening for audio...")

    from simulator.audio.mic_stream import MicSimulator

    try:
        while True:
            try:
                mic_mode = os.getenv("MIC_MODE", "sim")

                # --- Acquire signals ---
                try:
                    if mic_mode == "real":
                        from simulator.audio.real_mic import RealMic

                        device_id = int(os.getenv("MIC_DEVICE_ID", 0))
                        mic = RealMic(device_id=device_id)
                        signals, audio_paths = await mic.get_signals()

                    elif mic_mode == "real_3":
                        from simulator.audio.real_mic import RealMicArray

                        ids = [
                            int(x)
                            for x in os.getenv("MIC_DEVICE_IDS", "0,1,2").split(",")
                        ]
                        mic = RealMicArray(device_ids=ids)
                        signals, audio_paths = await mic.get_signals()

                    else:
                        scenario = os.getenv("DEMO_SCENARIO", "chainsaw")
                        mic_sim = MicSimulator(scenario)
                        signals, audio_paths = await mic_sim.get_signals()
                except Exception as e:
                    logger.error("Failed to acquire audio signals: %s", e)
                    await asyncio.sleep(5)
                    continue

                # --- Classify ---
                try:
                    audio = classify(audio_paths[0])
                except Exception as e:
                    logger.error("Classification failed: %s", e)
                    await asyncio.sleep(5)
                    continue

                logger.info("%s (%.0f%%)", audio.label, audio.confidence * 100)

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

                logger.info("%.4f°N %.4f°E", location.lat, location.lon)

                decision = decide(audio, location)
                logger.info("%s", decision.reason)

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

                    # --- LoRa send ---
                    packet = {
                        "class": audio.label,
                        "confidence": audio.confidence,
                        "lat": location.lat,
                        "lon": location.lon,
                        "priority": decision.priority,
                        "photo_b64": photo.b64 if photo else None,
                    }
                    try:
                        await asyncio.wait_for(lora.send(packet), timeout=10)
                        logger.info("LoRa packet sent to gateway")
                    except asyncio.TimeoutError:
                        logger.error("LoRa send timed out — packet lost")
                    except Exception as e:
                        logger.error("LoRa send failed — packet lost: %s", e)

                await asyncio.sleep(30)

            except Exception as e:
                logger.critical("Unexpected error in main loop: %s", e)
                await asyncio.sleep(10)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return


if __name__ == "__main__":
    asyncio.run(run())
