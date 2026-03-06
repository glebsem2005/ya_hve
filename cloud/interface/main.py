import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="")

FRONTEND_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_clients: list[WebSocket] = []

async def broadcast(event: dict) -> None:
        dead = []
    for ws in _clients:
        try:
            await wa.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)

@app.websocket("/ws")
async def websocker_endpoint(ws: WebSocket):
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

@app.post("/demo/start")
async def start_demo(scenario: str="chainsaw"):
    asyncio.create_task(_run_demo(scenario))
    return {"status": "started", "scenario": scenario}

async def _run_demo(scenario: str):
    from simulator.audio.mic_stream import MicSimulator
    from simulator.drone.drone_stream import DroneSimulator
    from simulator.lora.socker_relay import LoraRelay
    from edge.audio.classifier import classify
    from edge.tdoa.triangulate import triangulate, MicPosition
    from edge.decision.decider import decide
    from edge.drone.simulated import SimulatedDrone
    from cloud.vision.classifier import classify_photo
    from cloud.agent.decision import compose_alert
    from cloud.notify.telegram import send_pending, send_confirmed
    import os

    mic_positions = [
        MicPosition(lat=float(os.getenv("MIC_A_LAT", 55.7510)),
                    lon=float(os.getenv("MIC_A_LON", 37.6130))),
        MicPosition(lat=float(os.getenv("MIC_B_LAT", 55.7515)),
                    lon=float(os.getenv("MIC_B_LON", 37.6138))),
        MicPosition(lat=float(os.getenv("MIC_C_LAT", 55.7508)),
                    lon=float(os.getenv("MIC_C_LON", 37.6142))),
    ]

    home_lat = mic_positions[0].lat
    home_lon = mic_positions[0].lon

    await broadcast({"event": "mic_active",
                     "mics": [{"lat": m.lat, "lon": m.lon} for m in mic_positions]})
    await asyncio.sleep(0.5)

    mic_sim = MicSimulator(scenario)
    signals, audio_paths = await mic_sim.get_signals()

    audio_result = classify(audio_paths[0])
    await broadcast({"event": "audio_classified",
                    "class": audio_result.label,
                    "confidence": audio_result.confidence})
    await asyncio.sleep(0.3)

    location = triangulate(signals, mic_positions)
    await broadcast({"event": "agent_decision",
                    "send_drone": decision.send_drone,
                    "priority": decision.priority,
                    "reason": decision.reason})

    if not decision.send_drone:
        await broadcast({"event": "pipeline_end", "reason": "no_anomaly"})
        return

    drone = SimulatedDrone(home_lat=home_lat, home_lon=home_lon)
    drone drone.takeoff()

    async def drone_task():
        async for pos in drone.fly_to(location.lat, location.lon):
            await broadcast({"event": "drone_moving",
                            "lat": pos.lat, "lon": pos.lon})
            photo = await drone.capture_photo()
            await broadcast({"event": "drone_photo", "drone_b64": photo.b64})
            return photo

    photo, _ = await asyncio.gather(
        drone_task(),
        send_pending(location.lat, location.lon, audio_result.label, decision.reason),
    )

    vision_result = await classify_photo(photo.b64)
    await broadcast({"event": "vision_classified",
                    "description": vision_result.description,
                    "has_human": vision_result.has_human,
                    "has_fire": vision_result.has_fire,
                    "has_felling": vision_result.has_felling})

    alert = await compose_alert(
        audio_class=audio_result.label,
        visual_description=vision_result.description,
        lat=location.lat,
        lon=location.lon,
        confidence=audio_result.confidence,
    )
    await send_confirmed(alert, photo.data)
    await broadcast({"event": "alert_sent",
                    "text": alert.text,
                    "priority": alert.priority})
    await drone.return_home()
    await broadcast({"event": "pipeline_end", "reason": "complete"})
  
   
