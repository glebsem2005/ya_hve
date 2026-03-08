import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from cloud.notify.bot_app import start_bot, stop_bot

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    try:
        await start_bot()
    except Exception:
        logger.exception("Failed to start Telegram bot polling")
    yield
    await stop_bot()


app = FastAPI(title="ForestGuard", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_clients: list[WebSocket] = []


async def broadcast(event: dict) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _clients.remove(ws)


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text())


# ---- REST API v1 — designed for React/Flutter frontends ----


class DemoRequest(BaseModel):
    scenario: str = "chainsaw"


class DemoResponse(BaseModel):
    status: str
    scenario: str


@app.post("/api/v1/demo", response_model=DemoResponse)
async def start_demo_v1(req: DemoRequest):
    """Start a demo scenario. Future React/Flutter frontends use this endpoint."""
    asyncio.create_task(_run_demo(req.scenario))
    return DemoResponse(status="started", scenario=req.scenario)


# TODO: Future REST endpoints for React/Flutter frontend:
#
# GET  /api/v1/status          — current system status (mics, drone, alerts)
# GET  /api/v1/alerts          — list of past alerts with pagination
# GET  /api/v1/alerts/{id}     — single alert detail
# POST /api/v1/drone/dispatch  — manually dispatch drone to coordinates
# GET  /api/v1/drone/position  — current drone GPS position
# GET  /api/v1/mics            — list of mic positions and their status
# WS   /ws                     — real-time event stream (keep as-is)
#
# When building React/Flutter:
# 1. Use /api/v1/* endpoints for CRUD operations
# 2. Use /ws WebSocket for real-time map updates
# 3. Authentication: add OAuth2/JWT middleware to FastAPI
#    See: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/


# ---- Ranger management API ----

from cloud.db.rangers import (
    add_ranger,
    remove_ranger,
    get_all_rangers,
    get_ranger_by_chat_id,
    update_zone,
    set_active,
)


class RangerCreate(BaseModel):
    name: str
    chat_id: int
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float


class RangerZoneUpdate(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@app.get("/api/v1/rangers")
async def list_rangers():
    """List all registered rangers."""
    rangers = get_all_rangers()
    return [
        {
            "id": r.id,
            "name": r.name,
            "chat_id": r.chat_id,
            "zone": {
                "lat_min": r.zone_lat_min,
                "lat_max": r.zone_lat_max,
                "lon_min": r.zone_lon_min,
                "lon_max": r.zone_lon_max,
            },
            "active": r.active,
        }
        for r in rangers
    ]


@app.post("/api/v1/rangers")
async def create_ranger(req: RangerCreate):
    """Register a new ranger with their monitoring zone."""
    ranger = add_ranger(
        name=req.name,
        chat_id=req.chat_id,
        zone_lat_min=req.zone_lat_min,
        zone_lat_max=req.zone_lat_max,
        zone_lon_min=req.zone_lon_min,
        zone_lon_max=req.zone_lon_max,
    )
    return {"status": "created", "id": ranger.id, "name": ranger.name}


@app.delete("/api/v1/rangers/{chat_id}")
async def delete_ranger(chat_id: int):
    """Remove a ranger by their Telegram chat ID."""
    removed = remove_ranger(chat_id)
    if not removed:
        return {"status": "not_found"}
    return {"status": "removed"}


@app.patch("/api/v1/rangers/{chat_id}/zone")
async def update_ranger_zone(chat_id: int, req: RangerZoneUpdate):
    """Update a ranger's monitoring zone."""
    updated = update_zone(chat_id, req.lat_min, req.lat_max, req.lon_min, req.lon_max)
    if not updated:
        return {"status": "not_found"}
    return {"status": "updated"}


@app.patch("/api/v1/rangers/{chat_id}/active")
async def toggle_ranger_active(chat_id: int, active: bool = True):
    """Enable or disable alerts for a ranger."""
    updated = set_active(chat_id, active)
    if not updated:
        return {"status": "not_found"}
    return {"status": "active" if active else "inactive"}


# ---- Permits (лесные билеты) API ----

from datetime import date
from cloud.db.permits import (
    add_permit as db_add_permit,
    remove_permit as db_remove_permit,
    get_all_permits,
    get_permits_for_location,
    has_valid_permit,
)


class PermitCreate(BaseModel):
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float
    valid_from: date
    valid_until: date
    description: str = ""


class PermitCheck(BaseModel):
    lat: float
    lon: float


@app.get("/api/v1/permits")
async def list_permits():
    """List all logging permits."""
    permits = get_all_permits()
    return [
        {
            "id": p.id,
            "description": p.description,
            "zone": {
                "lat_min": p.zone_lat_min,
                "lat_max": p.zone_lat_max,
                "lon_min": p.zone_lon_min,
                "lon_max": p.zone_lon_max,
            },
            "valid_from": p.valid_from.isoformat(),
            "valid_until": p.valid_until.isoformat(),
        }
        for p in permits
    ]


@app.post("/api/v1/permits")
async def create_permit(req: PermitCreate):
    """Register a new logging permit."""
    permit = db_add_permit(
        zone_lat_min=req.zone_lat_min,
        zone_lat_max=req.zone_lat_max,
        zone_lon_min=req.zone_lon_min,
        zone_lon_max=req.zone_lon_max,
        valid_from=req.valid_from,
        valid_until=req.valid_until,
        description=req.description,
    )
    return {"status": "created", "id": permit.id}


@app.delete("/api/v1/permits/{permit_id}")
async def delete_permit(permit_id: int):
    """Remove a logging permit."""
    removed = db_remove_permit(permit_id)
    if not removed:
        return {"status": "not_found"}
    return {"status": "removed"}


@app.post("/api/v1/permits/check")
async def check_permit(req: PermitCheck):
    """Check if a location is covered by a valid permit."""
    has_permit = has_valid_permit(req.lat, req.lon)
    permits = get_permits_for_location(req.lat, req.lon)
    return {
        "has_valid_permit": has_permit,
        "permits": [{"id": p.id, "description": p.description} for p in permits],
    }


# ---- RAG query API ----

from cloud.agent.rag_agent import query_rag


class RagQueryRequest(BaseModel):
    question: str
    context: str = ""


class RagQueryResponse(BaseModel):
    answer: str


@app.post("/api/v1/rag-query", response_model=RagQueryResponse)
async def rag_query_endpoint(req: RagQueryRequest):
    """Query RAG agent with File Search + Web Search (Yandex AI Studio)."""
    answer = await query_rag(req.question, req.context)
    return RagQueryResponse(answer=answer)


# ---- Gateway event forwarding ----
# The LoRa gateway sends processed events (after Yandex GPT) here
# so they get broadcast to all web dashboard clients via WebSocket.


class GatewayEvent(BaseModel):
    event: str
    # Allow arbitrary extra fields for different event types
    model_config = {"extra": "allow"}


@app.post("/api/v1/gateway-event")
async def receive_gateway_event(payload: GatewayEvent):
    """Receive a processed event from the gateway and broadcast to dashboard."""
    await broadcast(payload.model_dump())
    return {"status": "broadcast"}


# Legacy endpoint for backward compatibility
@app.post("/demo/start")
async def start_demo_legacy(scenario: str = "chainsaw"):
    asyncio.create_task(_run_demo(scenario))
    return {"status": "started", "scenario": scenario}


async def _run_demo(scenario: str):
    from simulator.audio.mic_stream import MicSimulator
    from simulator.drone.drone_stream import DroneSimulator
    from simulator.lora.socket_relay import LoraRelay
    from edge.audio.classifier import classify
    from edge.audio.onset import OnsetDetector
    from edge.tdoa.triangulate import triangulate, MicPosition
    from edge.decision.decider import decide
    from edge.drone.simulated import SimulatedDrone
    from cloud.vision.classifier import classify_photo
    from cloud.agent.decision import compose_alert
    from cloud.notify.telegram import send_pending, send_confirmed
    import os

    mic_positions = [
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

    home_lat = mic_positions[0].lat
    home_lon = mic_positions[0].lon

    await broadcast(
        {
            "event": "mic_active",
            "mics": [{"lat": m.lat, "lon": m.lon} for m in mic_positions],
        }
    )
    await asyncio.sleep(0.5)

    mic_sim = MicSimulator(scenario)
    signals, audio_paths = await mic_sim.get_signals()

    # Onset detection — only proceed if sharp sound detected
    detector = OnsetDetector()
    onset = detector.detect(signals[0])
    await broadcast(
        {
            "event": "onset_check",
            "triggered": onset.triggered,
            "energy_ratio": round(onset.energy_ratio, 2),
        }
    )

    if not onset.triggered:
        await broadcast(
            {
                "event": "pipeline_end",
                "reason": f"no_onset (ratio={onset.energy_ratio:.1f})",
            }
        )
        return

    audio_result = classify(audio_paths[0])
    await broadcast(
        {
            "event": "audio_classified",
            "class": audio_result.label,
            "confidence": audio_result.confidence,
        }
    )
    await asyncio.sleep(0.3)

    location = triangulate(signals, mic_positions)

    decision = decide(audio_result, location)

    await broadcast(
        {
            "event": "location_found",
            "lat": location.lat,
            "lon": location.lon,
            "error_m": location.error_m,
        }
    )

    await broadcast(
        {
            "event": "agent_decision",
            "send_drone": decision.send_drone,
            "priority": decision.priority,
            "reason": decision.reason,
        }
    )

    if not decision.send_drone:
        await broadcast({"event": "pipeline_end", "reason": "no_anomaly"})
        return

    drone = SimulatedDrone(home_lat=home_lat, home_lon=home_lon)
    await drone.takeoff()

    async def drone_task():
        async for pos in drone.fly_to(location.lat, location.lon):
            await broadcast({"event": "drone_moving", "lat": pos.lat, "lon": pos.lon})
        photo = await drone.capture_photo()
        await broadcast({"event": "drone_photo", "drone_b64": photo.b64})
        return photo

    photo, _ = await asyncio.gather(
        drone_task(),
        send_pending(
            location.lat,
            location.lon,
            audio_result.label,
            decision.reason,
            confidence=audio_result.confidence,
        ),
    )

    vision_result = await classify_photo(photo.b64)
    await broadcast(
        {
            "event": "vision_classified",
            "description": vision_result.description,
            "has_human": vision_result.has_human,
            "has_fire": vision_result.has_fire,
            "has_felling": vision_result.has_felling,
        }
    )

    alert = await compose_alert(
        audio_class=audio_result.label,
        visual_description=vision_result.description,
        lat=location.lat,
        lon=location.lon,
        confidence=audio_result.confidence,
    )
    await send_confirmed(alert, photo.data)
    await broadcast(
        {"event": "alert_sent", "text": alert.text, "priority": alert.priority}
    )
    await drone.return_home()
    await broadcast({"event": "pipeline_end", "reason": "complete"})
