"""LoRa Gateway — receives edge packets, processes via Yandex GPT,
sends alerts to Telegram AND web dashboard.

Flow:
  Edge (onset → classify → triangulate) → LoRa → Gateway → Yandex GPT
    → Telegram (ranger)
    → Web dashboard (WebSocket broadcast)
"""

import asyncio
import json
import logging
import os

from cloud.vision.classifier import classify_photo
from cloud.agent.decision import compose_alert
from cloud.notify.telegram import send_pending, send_confirmed
from gateway.mesh import MeshRouter, MeshPacket

logger = logging.getLogger(__name__)

HOST = os.getenv("LORA_GATEWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("LORA_GATEWAY_PORT", 9000))
LORA_FREQ_MHZ = int(os.getenv("LORA_FREQ_MHZ", 868))
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://cloud:8000")

# Mesh router for deduplication of multi-hop packets
_mesh_router = MeshRouter()


async def _forward_to_dashboard(event: dict) -> None:
    """Forward a processed alert event to the cloud FastAPI server
    for WebSocket broadcast to the web dashboard."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{CLOUD_API_URL}/api/v1/gateway-event",
                json=event,
            )
    except Exception as e:
        print(f"Failed to forward event to dashboard: {e}")


async def handle_packet(packet: dict) -> None:

    # Mesh deduplication: if packet has mesh headers, check for duplicates
    if "packet_id" in packet:
        mesh_pkt = MeshPacket(
            packet_id=packet["packet_id"],
            source_node=packet.get("source_node", "unknown"),
            hop_count=packet.get("hop_count", 0),
            max_hops=packet.get("max_hops", 3),
            route=packet.get("route", []),
            payload=packet,
        )
        payload = _mesh_router.process_packet(mesh_pkt)
        if payload is None:
            logger.info("Mesh: duplicate packet dropped")
            return

    logger.info("Packet received from edge:")
    logger.info("class=%s  conf=%.0f%%", packet["class"], packet["confidence"] * 100)
    logger.info("lat=%.4f  lon=%.4f", packet["lat"], packet["lon"])

    # Notify dashboard: sound detected and localized
    await _forward_to_dashboard(
        {
            "event": "audio_classified",
            "class": packet["class"],
            "confidence": packet["confidence"],
        }
    )
    await _forward_to_dashboard(
        {
            "event": "location_found",
            "lat": packet["lat"],
            "lon": packet["lon"],
            "error_m": packet.get("error_m", 0.0),
        }
    )

    photo_b64 = packet.get("photo_b64", "")

    if photo_b64:
        print("Sending photo to YandexGPT Vision...")
        vision = await classify_photo(photo_b64)
        print(f"   Vision: {vision.description}")

        await _forward_to_dashboard(
            {
                "event": "vision_classified",
                "description": vision.description,
                "has_human": vision.has_human,
                "has_fire": vision.has_fire,
                "has_felling": vision.has_felling,
            }
        )
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

    # Classification agent verification (non-blocking, enriches context)
    try:
        from cloud.agent.classification_agent import verify_classification

        verification = await verify_classification(
            audio_class=packet["class"],
            confidence=packet["confidence"],
            lat=packet["lat"],
            lon=packet["lon"],
            ndsi=packet.get("ndsi"),
        )
        logger.info("Agent verification: priority=%s", verification.priority)
        await _forward_to_dashboard(
            {
                "event": "agent_verified",
                "priority": verification.priority,
                "context_analysis": verification.context_analysis,
                "recommended_action": verification.recommended_action,
            }
        )
    except Exception as e:
        logger.warning("Classification agent failed: %s", e)

    # Send pending alert first (creates Incident with accept button)
    incident = await send_pending(
        lat=packet["lat"],
        lon=packet["lon"],
        audio_class=packet["class"],
        reason="gateway detection",
        confidence=packet["confidence"],
    )

    # Store drone photo in incident (will be sent after ranger accepts)
    photo_bytes = None
    if photo_b64:
        import base64

        photo_bytes = base64.b64decode(photo_b64)

    await send_confirmed(alert, photo_bytes, incident=incident)
    print("Alert sent to Telegram.")

    # Forward final alert to web dashboard
    await _forward_to_dashboard(
        {
            "event": "alert_sent",
            "text": alert.text,
            "priority": alert.priority,
        }
    )
    await _forward_to_dashboard(
        {
            "event": "pipeline_end",
            "reason": "complete",
        }
    )
    print("Alert forwarded to web dashboard.\n")


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
    logger.info(
        "LoRa Gateway listening on %s:%d (radio: %d MHz)", HOST, PORT, LORA_FREQ_MHZ
    )
    logger.info("Waiting for packets from edge server...")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
