"""
main.py — FastAPI backend entry point for MARL City Simulator.

Provides the WebSocket API for the frontend and HTTP endpoints
for configuration and exports. Uses a background task to run the
simulation loop asynchronously while keeping the API responsive.

API Endpoints:
  - GET  /                     : Health check
  - GET  /api/config           : Get simulation configuration
  - POST /api/export/metrics   : Export metrics to CSV
  - WS   /ws                   : Main simulation websocket

WebSocket Protocol (Client to Server):
  - {"action": "speed", "value": "play"}      : Set speed
  - {"action": "skip", "value": 30}           : Skip steps
  - {"action": "inspect", "agent_id": 42}     : Get agent details
  - {"action": "policy", "text": "UBI..."}    : Apply policy
  - {"action": "compare", "text": "UBI...", "steps": 90} : Run A/B comparison

WebSocket Protocol (Server to Client):
  - {"type": "full_state", ...}               : Initial sync state
  - {"type": "diff", ...}                     : Timestep diff (position/state)
  - {"type": "inspect_result", ...}           : Requested agent data
  - {"type": "policy_result", ...}            : Policy application result
  - {"type": "comparison_result", ...}        : A/B scenario results

Author: Aditya Padale (B.Tech Final Year Project)
"""

import asyncio
import json
import logging
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from config import FASTAPI_HOST, FASTAPI_PORT, ALLOWED_ORIGINS
from simulation import SimulationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("marl_server")

# Create FastAPI app
app = FastAPI(
    title="MARL City Simulator API",
    description="Backend for the visual agent-based economy simulator",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
sim: SimulationEngine = None
sim_task: asyncio.Task = None

# Active WebSocket connections
active_connections: set[WebSocket] = set()

# ═══════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize the simulation engine on server start."""
    global sim, sim_task
    logger.info("Initializing MARL City Simulator...")

    # Create and initialize simulation
    sim = SimulationEngine()
    sim.initialize()

    # Start the async simulation loop
    # We pass broadcast_state as the callback to emit diffs to all clients
    sim_task = asyncio.create_task(sim.run_loop(callback=broadcast_state))
    logger.info("Simulation engine started.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown."""
    global sim, sim_task
    logger.info("Shutting down MARL City Simulator...")

    if sim:
        sim.stop()
    if sim_task:
        sim_task.cancel()

    # Close all active connections
    for ws in active_connections:
        await ws.close()

    logger.info("Shutdown complete.")

# ═══════════════════════════════════════════
# WEBSOCKET BROADCAST
# ═══════════════════════════════════════════

async def broadcast_state(diff: Dict[str, Any]):
    """
    Broadcast a state diff to all connected WebSocket clients.
    Called by the simulation loop every tick.
    """
    if not active_connections:
        return

    # Serialize once
    diff_json = json.dumps(diff)

    # Send to all connected clients
    disconnected = set()
    for ws in active_connections:
        try:
            await ws.send_text(diff_json)
        except Exception:
            disconnected.add(ws)

    # Clean up dead connections
    for ws in disconnected:
        active_connections.remove(ws)

# ═══════════════════════════════════════════
# HTTP ENDPOINTS
# ═══════════════════════════════════════════

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "simulation_running": sim.is_running if sim else False,
        "timestep": sim.timestep if sim else 0,
        "active_clients": len(active_connections)
    }

@app.get("/api/config")
async def get_config():
    """Get simulation configuration for frontend rendering."""
    from config import CANVAS_WIDTH, CANVAS_HEIGHT, AGENT_TYPE_NAMES
    return {
        "canvas": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
        },
        "agent_types": AGENT_TYPE_NAMES,
    }

class ExportRequest(BaseModel):
    filename: str = "data/metrics_export.csv"

@app.post("/api/export/metrics")
async def export_metrics(req: ExportRequest):
    """Export metrics to CSV."""
    if not sim:
        raise HTTPException(status_code=500, detail="Simulation not running")
    try:
        sim.export_metrics(req.filename)
        return {"status": "success", "file": req.filename}
    except Exception as e:
        logger.error("Export failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket connection for the frontend.
    Handles initial state sync and incoming client commands.
    """
    await websocket.accept()
    active_connections.add(websocket)
    logger.info("New WebSocket client connected. Active: %d", len(active_connections))

    try:
        # 1. Send full initial state
        if sim:
            full_state = sim.get_full_state()
            await websocket.send_json(full_state)

        # 2. Listen for client commands
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                await process_command(command, websocket)
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON from client: %s", data)
            except Exception as e:
                logger.error("Error processing client command: %s", e)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        if websocket in active_connections:
            active_connections.remove(websocket)

async def process_command(cmd: Dict[str, Any], websocket: WebSocket):
    """Process a command received from a WebSocket client."""
    action = cmd.get("action")

    if not action or not sim:
        return

    # ── Speed Control ──
    if action == "speed":
        speed = cmd.get("value", "play")
        sim.set_speed(speed)

    # ── Time Skip ──
    elif action == "skip":
        steps = int(cmd.get("value", 30))
        sim.skip_steps(steps)
        # Send full state sync after skip
        await websocket.send_json(sim.get_full_state())

    # ── Agent Inspection ──
    elif action == "inspect":
        agent_id = int(cmd.get("agent_id", 0))
        inspect_data = sim.get_agent_inspect(agent_id)
        if inspect_data:
            await websocket.send_json({
                "type": "inspect_result",
                "data": inspect_data
            })

    # ── Policy Application ──
    elif action == "policy":
        text = cmd.get("text", "")
        if text:
            # Temporarily pause simulation while Gemini processes
            was_paused = sim.is_paused
            prev_speed = sim.speed
            sim.set_speed("pause")

            try:
                result = await sim.apply_policy_text(text)
                await websocket.send_json({
                    "type": "policy_result",
                    "success": True,
                    "data": result
                })
                # Resend full state as many things might have changed
                await websocket.send_json(sim.get_full_state())
            except Exception as e:
                logger.error("Policy application failed: %s", e)
                await websocket.send_json({
                    "type": "policy_result",
                    "success": False,
                    "error": str(e)
                })
            finally:
                # Restore speed
                if not was_paused:
                    sim.set_speed(prev_speed)

    # ── Scenario Comparison (A/B Test) ──
    elif action == "compare":
        text = cmd.get("text", "")
        steps = int(cmd.get("steps", 90))

        if text:
            was_paused = sim.is_paused
            prev_speed = sim.speed
            sim.set_speed("pause")

            try:
                # Let frontend know comparison is starting
                await websocket.send_json({
                    "type": "comparison_started",
                    "steps": steps
                })

                # Run comparison (this takes a few seconds)
                result = await sim.run_comparison(text, steps)

                # Send results back
                await websocket.send_json({
                    "type": "comparison_result",
                    "data": result
                })
                # Resend full state
                await websocket.send_json(sim.get_full_state())
            except Exception as e:
                logger.error("Comparison failed: %s", e)
                await websocket.send_json({
                    "type": "comparison_error",
                    "error": str(e)
                })
            finally:
                if not was_paused:
                    sim.set_speed(prev_speed)

    # ── Reset Simulation ──
    elif action == "reset":
        logger.info("Resetting simulation...")
        sim.stop()
        sim.initialize()
        sim.set_speed("play")
        await websocket.send_json(sim.get_full_state())

# ═══════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MARL City Simulator on %s:%d", FASTAPI_HOST, FASTAPI_PORT)
    uvicorn.run("main:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=True)
