import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cloud.vision.classifier import classify_photo
from cloud.agent.decision import compose_alert
from cloud.notify.telegram import send_confirmed

HOST = os.getenv("LORA_GATEWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("LORA_GATEWAY_PORT", 9000))

async def handle_packet(packet: dict) -> None:

    print(f"\nPacket received from edge:")
    print(f"class={packet['class']}  conf={packet['confidence']:.0%}")
    print(f"lat={packet['lat']:.4f}  lon={packet['lon']:.4f}")

    photo_b64 = packet.get("photo_b64", "")

    if photo_b64:
        print("Sending photo to YandexGPT Vision...")
        vision = await classify_photo(photo_b64)
        print(f"   Vision: {vision.description}")
    else:
        from cloud.vision.classifier import VisionResult
        vision = VisionResult(
            description="Фото недоступно",
            has_human=False,
            has_fire=False,
            has_felling=False,
        )
      
    print("Composing alert via YandexGPT...")
    alert = await compose_alert(
        audio_class=packet["class"],
        visual_description=vision.description,
        lat=packet["lat"],
        lon=packet["lon"],
        confidence=packet["confidence"],
    )
    print(f"   Alert: {alert.text[:80]}...")

    print("Sending Telegram alert to ranger...")
    photo_bytes = None
    if photo_b64:
        import base64
        photo_bytes = base64.b64decode(photo_b64)

    await send_confirmed(alert, photo_bytes)
    print("Alert sent.\n")


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        data = await reader.readline()
        if data:
            packet = json.loads(data.decode().strip())
            await handle_packet(packet)
    except json.JSONDecodeError as e:
        print(f"Invalid packet: {e}")
    except Exception as e:
        print(f"Error processing packet: {e}")
    finally:
        writer.close()


async def main():
    server = await asyncio.start_server(handle_connection, HOST, PORT)
    print(f"LoRa Gateway listening on {HOST}:{PORT}")
    print("Waiting for packets from edge server...\n")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
