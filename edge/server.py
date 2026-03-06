import asyncio
import os
from edge.audio.classifier import classify
from edge.tdoa.triangulate import triangulate, MicPosition
from edge.decision.decider import decide
from edge.drone.simulated import SimulatedDrone
from simulator.lora.socket_relay import LoraRelay

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

    print("🖥  Edge server started — listening for audio...")

    from simulator.audio.mic_stream import MicSimulator

    while True:
        # Switch via .env:
        # MIC_MODE=sim      -- MicSimulator (default, no hardware needed)
        # MIC_MODE=real     -- RealMic (one USB mic plugged in)
        # MIC_MODE=real_3   -- RealMicArray (3 USB mics for real TDOA)
        mic_mode = os.getenv("MIC_MODE", "sim")

        if mic_mode == "real":
            from simulator.audio.real_mic import RealMic

            device_id = int(os.getenv("MIC_DEVICE_ID", 0))
            mic = RealMic(device_id=device_id)
            signals, audio_paths = await mic.get_signals()

        elif mic_mode == "real_3":
            from simulator.audio.real_mic import RealMicArray

            ids = [int(x) for x in os.getenv("MIC_DEVICE_IDS", "0,1,2").split(",")]
            mic = RealMicArray(device_ids=ids)
            signals, audio_paths = await mic.get_signals()

        else:
            scenario = os.getenv("DEMO_SCENARIO", "chainsaw")
            mic_sim = MicSimulator(scenario)
            signals, audio_paths = await mic_sim.get_signals()

        audio = classify(audio_paths[0])
        print(f"{audio.label} ({audio.confidence:.0%})")

        location = triangulate(signals, MIC_POSITIONS)
        print(f"{location.lat:.4f}°N {location.lon:.4f}°E")

        decision = decide(audio, location)
        print(f"{decision.reason}")

        if decision.send_drone:
            print("🚁 Launching drone...")
            await drone.takeoff()
            async for pos in drone.fly_to(location.lat, location.lon):
                # In real deployment: broadcast position over LoRa back to gateway
                print(f"   🚁 {pos.lat:.4f}°N {pos.lon:.4f}°E")
            photo = await drone.capture_photo()
            await drone.return_home()

            # Send LoRa packet to gateway
            packet = {
                "class": audio.label,
                "confidence": audio.confidence,
                "lat": location.lat,
                "lon": location.lon,
                "priority": decision.priority,
                "photo_b64": photo.b64,
            }
            await lora.send(packet)
            print("LoRa packet sent to gateway")

        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(run())
