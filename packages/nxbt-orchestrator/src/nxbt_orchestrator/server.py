"""FastAPI app exposing the nxbt orchestrator over HTTP + WebSocket.

Endpoints:

  ``GET  /health``         — liveness + connection probe.
  ``GET  /state``          — one-shot snapshot of current controller state.
  ``POST /action``         — apply one frame; accepts a ``Packet`` JSON or a
                              26-dim float vector. Optional ``source`` field
                              gates inference traffic via the human-override
                              window.
  ``POST /buttons``        — list of held button names (human input).
  ``POST /stick``          — analog stick update (human input).
  ``POST /macro``          — fire an nxbt macro string.
  ``POST /control``        — toggle recording, set recording output path.
  ``WS   /ws/state``       — live state stream at the update rate.

The :class:`NxbtController` is created at app startup and stopped on
shutdown.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from nx_packets import ACTION_DIM, Packet

from nxbt_orchestrator.controller import ActionSource, NxbtController
from nxbt_orchestrator.state_stream import StateStream


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7777
    update_rate: int = 120
    override_window: float = 0.3
    reconnect_address: str | None = None
    debug: bool = False
    recording_output_path: str = "recorded_macro.json"


class ActionRequest(BaseModel):
    """Either ``packet`` or ``vector`` must be provided, not both."""

    packet: Packet | None = None
    vector: list[float] | None = Field(default=None, min_length=ACTION_DIM, max_length=ACTION_DIM)
    source: ActionSource = "inference"
    button_threshold: float = 0.5


class ButtonsRequest(BaseModel):
    buttons: list[str]
    source: ActionSource = "human"


class StickRequest(BaseModel):
    stick: Literal["LEFT_STICK", "RIGHT_STICK"]
    x: int
    y: int
    source: ActionSource = "human"


class MacroRequest(BaseModel):
    macro: str
    block: bool = False


class ControlRequest(BaseModel):
    command: Literal["toggle_recording", "set_output_path"]
    path: str | None = None


def create_app(config: ServerConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        controller = NxbtController(
            reconnect_address=config.reconnect_address,
            update_rate=config.update_rate,
            override_window=config.override_window,
            recording_output_path=config.recording_output_path,
            debug=config.debug,
        )
        await asyncio.to_thread(controller.start)
        app.state.controller = controller
        app.state.stream = StateStream(controller)
        try:
            yield
        finally:
            controller.stop()

    app = FastAPI(title="nxbt-orchestrator", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        c: NxbtController = app.state.controller
        return {
            "running": c.is_running,
            "connected": c.is_connected,
            "update_rate": c.update_rate,
            "override_window": c.override_window,
            "recording": c.recording_active,
        }

    @app.get("/state")
    async def get_state() -> dict[str, Any]:
        c: NxbtController = app.state.controller
        return c.snapshot_state()

    @app.post("/action")
    async def post_action(req: ActionRequest) -> dict[str, Any]:
        if (req.packet is None) == (req.vector is None):
            raise HTTPException(
                status_code=400,
                detail="exactly one of `packet` or `vector` is required",
            )
        c: NxbtController = app.state.controller
        if req.packet is not None:
            applied = c.apply_packet(req.packet, source=req.source)
        else:
            assert req.vector is not None
            applied = c.apply_action_vector(
                req.vector,
                source=req.source,
                button_threshold=req.button_threshold,
            )
        return {"applied": applied}

    @app.post("/buttons")
    async def post_buttons(req: ButtonsRequest) -> dict[str, Any]:
        c: NxbtController = app.state.controller
        c.apply_button_set(req.buttons, source=req.source)
        return {"applied": True}

    @app.post("/stick")
    async def post_stick(req: StickRequest) -> dict[str, Any]:
        c: NxbtController = app.state.controller
        c.apply_stick(req.stick, req.x, req.y, source=req.source)
        return {"applied": True}

    @app.post("/macro")
    async def post_macro(req: MacroRequest) -> dict[str, Any]:
        c: NxbtController = app.state.controller
        macro_id = c.fire_macro(req.macro, block=req.block)
        return {"macro_id": macro_id}

    @app.post("/control")
    async def post_control(req: ControlRequest) -> dict[str, Any]:
        c: NxbtController = app.state.controller
        if req.command == "toggle_recording":
            return {"recording": c.toggle_recording()}
        if req.command == "set_output_path":
            if not req.path:
                raise HTTPException(status_code=400, detail="`path` is required")
            c.set_recording_path(req.path)
            return {"path": req.path}
        raise HTTPException(status_code=400, detail=f"unknown command: {req.command}")

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket) -> None:
        await ws.accept()
        stream: StateStream = app.state.stream
        sub = stream.subscribe()
        try:
            async for snapshot in sub:
                await ws.send_text(json.dumps(snapshot))
        except WebSocketDisconnect:
            pass
        finally:
            sub.close()

    return app
