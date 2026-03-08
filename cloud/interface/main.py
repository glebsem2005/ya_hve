import asyncio
import json
import logging
import math
import random
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from cloud.notify.bot_app import start_bot, stop_bot

logger = logging.getLogger(__name__)


def _random_source_point(
    mic_positions: list[tuple[float, float]], radius_m: float = 150.0
) -> tuple[float, float]:
    """Generate a random sound source near the centroid of mic positions."""
    avg_lat = sum(p[0] for p in mic_positions) / len(mic_positions)
    avg_lon = sum(p[1] for p in mic_positions) / len(mic_positions)
    # Random distance 50-300m in random direction
    dist = random.uniform(50.0, min(radius_m * 2, 300.0))
    bearing = random.uniform(0, 2 * math.pi)
    # Approximate degree offsets
    dlat = (dist * math.cos(bearing)) / 111_320
    dlon = (dist * math.sin(bearing)) / (111_320 * math.cos(math.radians(avg_lat)))
    return round(avg_lat + dlat, 6), round(avg_lon + dlon, 6)


async def _auto_demo():
    """Auto-start a demo scenario after container boot."""
    delay = random.uniform(15, 25)
    logger.info("Auto-demo scheduled in %.0f seconds", delay)
    await asyncio.sleep(delay)

    from cloud.db.microphones import get_online

    online_mics = get_online()[:3]
    if len(online_mics) < 3:
        logger.warning("Auto-demo: need 3 online mics, got %d", len(online_mics))
        return

    mic_coords = [(m.lat, m.lon) for m in online_mics]
    source_lat, source_lon = _random_source_point(mic_coords)
    scenario = random.choice(["chainsaw", "gunshot", "engine"])

    logger.info("Auto-demo: %s at (%.4f, %.4f)", scenario, source_lat, source_lon)
    try:
        await _run_demo(scenario, source_lat=source_lat, source_lon=source_lon)
    except Exception:
        logger.exception("Auto-demo failed")


@asynccontextmanager
async def lifespan(app):
    try:
        await start_bot()
    except Exception:
        logger.exception("Failed to start Telegram bot polling")
    asyncio.create_task(_auto_demo())
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
    source_lat: float | None = None
    source_lon: float | None = None


class DemoResponse(BaseModel):
    status: str
    scenario: str


@app.post("/api/v1/demo", response_model=DemoResponse)
async def start_demo_v1(req: DemoRequest):
    """Start a demo scenario. Optionally specify source coordinates."""
    asyncio.create_task(
        _run_demo(req.scenario, source_lat=req.source_lat, source_lon=req.source_lon)
    )
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
            "current_lat": r.current_lat,
            "current_lon": r.current_lon,
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


# ---- DataSphere cloud classification (2-tier) ----

from cloud.agent.datasphere_client import classify_embeddings


class ClassifyRequest(BaseModel):
    embeddings: list[float]


@app.post("/api/v1/classify")
async def classify_cloud(req: ClassifyRequest):
    """Cloud classification via DataSphere Node (2-tier verification)."""
    result = await classify_embeddings(req.embeddings)
    if result is None:
        return {"status": "unavailable", "message": "DataSphere Node not configured"}
    return {"status": "ok", **result}


# ---- DataLens: incidents export ----

from fastapi.responses import PlainTextResponse
from cloud.analytics.sample_incidents import get_incidents_csv_text


@app.get("/api/v1/incidents/export", response_class=PlainTextResponse)
async def export_incidents_csv():
    """Export incidents as CSV for DataLens integration."""
    return PlainTextResponse(
        content=get_incidents_csv_text(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=incidents.csv"},
    )


# ---- AI Studio stack info ----


@app.get("/api/v1/ai-studio-stack")
async def ai_studio_stack():
    """Show all Yandex Cloud AI Studio integrations used."""
    return {
        "integrations": [
            {
                "service": "YandexGPT",
                "usage": "Alert composition, legal text generation",
            },
            {
                "service": "AI Studio Assistants API",
                "usage": "RAG agent with File Search",
            },
            {
                "service": "File Search (RAG)",
                "usage": "Legal knowledge base (9 normative docs)",
            },
            {
                "service": "Web Search",
                "usage": "Real-time legal updates from consultant.ru/garant.ru",
            },
            {"service": "SpeechKit STT", "usage": "Voice message transcription"},
            {"service": "YandexGPT Vision", "usage": "Drone photo analysis"},
            {
                "service": "DataSphere",
                "usage": "ML model training & deployment (YAMNet)",
            },
            {"service": "DataLens", "usage": "Analytics dashboard for management"},
            {
                "service": "Yandex Workflows",
                "usage": "Incident processing pipeline orchestration",
            },
        ]
    }


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


# ---- Microphone network API ----

from cloud.db.microphones import (
    seed_microphones,
    get_all as mic_get_all,
    get_online as mic_get_online,
    get_by_uid as mic_get_by_uid,
    set_status as mic_set_status,
    set_battery as mic_set_battery,
)

# Seed microphones on startup
seed_microphones()


class MicStatusUpdate(BaseModel):
    status: str  # online, offline, broken


class MicBatteryUpdate(BaseModel):
    battery_pct: float


@app.get("/api/v1/mics")
async def list_mics():
    """List all microphones in the network."""
    mics = mic_get_all()
    return [
        {
            "mic_uid": m.mic_uid,
            "lat": m.lat,
            "lon": m.lon,
            "zone_type": m.zone_type,
            "sub_district": m.sub_district,
            "status": m.status,
            "battery_pct": m.battery_pct,
            "installed_at": m.installed_at,
        }
        for m in mics
    ]


@app.get("/api/v1/mics/online")
async def list_mics_online():
    """List only online microphones."""
    mics = mic_get_online()
    return [
        {"mic_uid": m.mic_uid, "lat": m.lat, "lon": m.lon, "zone_type": m.zone_type}
        for m in mics
    ]


@app.patch("/api/v1/mics/{mic_uid}/status")
async def update_mic_status(mic_uid: str, req: MicStatusUpdate):
    """Update microphone status."""
    ok = mic_set_status(mic_uid, req.status)
    if not ok:
        return {"status": "not_found"}
    return {"status": "updated", "mic_uid": mic_uid, "new_status": req.status}


@app.patch("/api/v1/mics/{mic_uid}/battery")
async def update_mic_battery(mic_uid: str, req: MicBatteryUpdate):
    """Update microphone battery percentage."""
    ok = mic_set_battery(mic_uid, req.battery_pct)
    if not ok:
        return {"status": "not_found"}
    return {"status": "updated", "mic_uid": mic_uid, "battery_pct": req.battery_pct}


# ---- DataLens JSON endpoints ----

from cloud.analytics.datalens import get_datalens_incidents, get_datalens_stats


@app.get("/api/v1/datalens/incidents")
async def datalens_incidents():
    """JSON incidents data for DataLens API connector."""
    return get_datalens_incidents()


@app.get("/api/v1/datalens/stats")
async def datalens_stats():
    """Aggregated statistics for DataLens dashboard."""
    return get_datalens_stats()


# ---- Classification agent ----

from cloud.agent.classification_agent import verify_classification


class ClassifyAgentRequest(BaseModel):
    audio_class: str
    confidence: float
    lat: float
    lon: float
    zone_type: str = "exploitation"
    ndsi: float | None = None


@app.post("/api/v1/agent/classify")
async def classify_agent(req: ClassifyAgentRequest):
    """Real-time classification verification via AI Studio agent."""
    result = await verify_classification(
        audio_class=req.audio_class,
        confidence=req.confidence,
        lat=req.lat,
        lon=req.lon,
        zone_type=req.zone_type,
        ndsi=req.ndsi,
    )
    return {
        "verified_class": result.verified_class,
        "confidence": result.confidence,
        "priority": result.priority,
        "context_analysis": result.context_analysis,
        "recommended_action": result.recommended_action,
        "permit_status": result.permit_status,
    }


# ---- FGIS-LK endpoints (stub) ----

from cloud.integrations.fgis_lk import fgis_client, ViolationReport


class ViolationSubmit(BaseModel):
    incident_id: str
    audio_class: str
    lat: float
    lon: float
    confidence: float
    ranger_name: str = ""
    description: str = ""


@app.get("/api/v1/fgis-lk/forest-unit")
async def fgis_forest_unit(lat: float, lon: float):
    """Look up forest quarter by coordinates (FGIS-LK stub)."""
    unit = fgis_client.get_forest_unit(lat, lon)
    return {
        "quarter_number": unit.quarter_number,
        "sub_district": unit.sub_district,
        "species_composition": unit.species_composition,
        "zone_type": unit.zone_type,
        "area_ha": unit.area_ha,
    }


@app.get("/api/v1/fgis-lk/permits")
async def fgis_permits(lat: float, lon: float):
    """Get active felling permits for location (FGIS-LK stub)."""
    permits = fgis_client.get_active_permits(lat, lon)
    return [
        {
            "permit_id": p.permit_id,
            "felling_type": p.felling_type,
            "volume_m3": p.volume_m3,
            "contractor": p.contractor,
            "valid_from": p.valid_from.isoformat(),
            "valid_until": p.valid_until.isoformat(),
        }
        for p in permits
    ]


@app.post("/api/v1/fgis-lk/violation")
async def fgis_violation(req: ViolationSubmit):
    """Submit violation report to FGIS-LK (stub)."""
    report = ViolationReport(
        incident_id=req.incident_id,
        audio_class=req.audio_class,
        lat=req.lat,
        lon=req.lon,
        confidence=req.confidence,
        ranger_name=req.ranger_name,
        description=req.description,
        timestamp="",
    )
    result = fgis_client.submit_violation(report)
    return result


# ---- Workflow endpoints ----

from cloud.workflows.pipeline import get_pipeline_definition
from cloud.workflows.yandex_workflows import register_workflow, run_workflow


@app.get("/api/v1/workflow/definition")
async def workflow_definition():
    """Return the full pipeline definition as JSON."""
    return get_pipeline_definition()


class WorkflowRunRequest(BaseModel):
    scenario: str = "chainsaw"


@app.post("/api/v1/workflow/run")
async def workflow_run(req: WorkflowRunRequest):
    """Run the incident processing pipeline via WorkflowExecutor."""
    reg = await register_workflow()
    result = await run_workflow(reg["workflow_id"], {"scenario": req.scenario})
    # Also trigger the demo pipeline
    asyncio.create_task(_run_demo(req.scenario))
    return result


# Legacy endpoint for backward compatibility
@app.post("/demo/start")
async def start_demo_legacy(scenario: str = "chainsaw"):
    asyncio.create_task(_run_demo(scenario))
    return {"status": "started", "scenario": scenario}


async def _run_demo(
    scenario: str,
    source_lat: float | None = None,
    source_lon: float | None = None,
):
    from simulator.audio.mic_stream import MicSimulator
    from simulator.drone.drone_stream import DroneSimulator
    from simulator.lora.socket_relay import LoraRelay
    from edge.audio.classifier import classify
    from edge.audio.onset import detect_onset as detect_onset_fn
    from edge.tdoa.triangulate import triangulate, MicPosition
    from edge.decision.decider import decide
    from edge.drone.simulated import SimulatedDrone
    from cloud.vision.classifier import classify_photo
    from cloud.agent.decision import compose_alert
    from cloud.notify.telegram import send_pending, send_confirmed
    from cloud.db.microphones import get_online
    import os

    # Read mic positions from DB (first 3 online), fallback to env vars
    online_mics = get_online()[:3]
    if len(online_mics) >= 3:
        mic_positions = [MicPosition(lat=m.lat, lon=m.lon) for m in online_mics]
    else:
        mic_positions = [
            MicPosition(
                lat=float(os.getenv("MIC_A_LAT", 57.3697)),
                lon=float(os.getenv("MIC_A_LON", 44.6200)),
            ),
            MicPosition(
                lat=float(os.getenv("MIC_B_LAT", 57.3752)),
                lon=float(os.getenv("MIC_B_LON", 44.6345)),
            ),
            MicPosition(
                lat=float(os.getenv("MIC_C_LAT", 57.3631)),
                lon=float(os.getenv("MIC_C_LON", 44.6489)),
            ),
        ]

    mic_coords = [(m.lat, m.lon) for m in mic_positions]

    # Generate random source if not specified
    if source_lat is None or source_lon is None:
        source_lat, source_lon = _random_source_point(mic_coords)

    home_lat = mic_positions[0].lat
    home_lon = mic_positions[0].lon

    await broadcast(
        {
            "event": "mic_active",
            "mics": [{"lat": m.lat, "lon": m.lon} for m in mic_positions],
        }
    )
    await broadcast(
        {
            "event": "source_point",
            "lat": source_lat,
            "lon": source_lon,
            "scenario": scenario,
        }
    )
    await asyncio.sleep(0.5)

    mic_sim = MicSimulator(
        scenario,
        source_lat=source_lat,
        source_lon=source_lon,
        mic_positions=mic_coords,
    )
    signals, audio_paths = await mic_sim.get_signals()

    # Onset detection — pre-filled quiet baseline so gunshot/engine also trigger
    onset = detect_onset_fn(signals[0])
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

    # send_pending creates an Incident and returns it
    photo, incident = await asyncio.gather(
        drone_task(),
        send_pending(
            location.lat,
            location.lon,
            audio_result.label,
            decision.reason,
            confidence=audio_result.confidence,
            is_demo=True,
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
    # Store drone photo in incident (sent to ranger after accept)
    await send_confirmed(alert, photo.data, incident=incident)
    await broadcast(
        {"event": "alert_sent", "text": alert.text, "priority": alert.priority}
    )
    await drone.return_home()
    await broadcast({"event": "pipeline_end", "reason": "complete"})
