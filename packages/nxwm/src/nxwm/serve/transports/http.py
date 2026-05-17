"""HTTP/WebSocket transport for :class:`WorldModelServer`.

Serves the world model over HTTPS so the UI (and any other client) can
talk to it the same way it talks to a ``zmq://`` endpoint. The hot per-frame
loop runs over a WebSocket — action in, JPEG (+ telemetry) out — so the
overhead of an HTTP round-trip per 33 ms is avoided. Session state is
per-connection: each WS opens its own :class:`Session` on the server.

Endpoints (all under ``/v1``):

  =======  ==================  =====================================
  GET      ``/v1/info``         JSON snapshot (matches ZMQ info wire)
  POST     ``/v1/reload``       ``{"model_path": str}`` → ``{"ok": true}``
                                 (admin op — affects all live sessions)
  GET      ``/v1/episodes``     ``{"episodes": [...]}``
  WS       ``/v1/session``      see "WebSocket protocol" below
  =======  ==================  =====================================

WebSocket protocol
------------------

On open the server sends a JSON text frame::

    {"type": "ready",
     "history_length": int,
     "goal_offset": int,
     "action_dim": int,
     "current_episode_file": str | None}

Client→server frames:

* **Binary, exactly 104 bytes** — a 26-float32 action. Server replies with
  one text frame (telemetry, may be ``{}``) followed by one binary frame
  (JPEG bytes). On error: one text frame ``{"error": "..."}`` and no JPEG.
* **Binary, len > 104, first byte 0x01** — ``MODE_INIT`` payload
  (see :mod:`nxwm.serve.transports.zmq._parse_init_payload`). Server
  reseeds and replies with ``{"type": "ok"}``.
* **Text frame, JSON** — a control op:

  - ``{"op": "reseed", "file": str, "start_frame": int}``
  - ``{"op": "detector_config"}`` → returns config
  - ``{"op": "detector_params", "params": {...}}``
  - ``{"op": "detector_reset"}``
  - ``{"op": "info"}``

  Server replies with one text frame whose shape depends on the op
  (``{"type": "ok"}``, or the requested payload, or ``{"error": "..."}``).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from nxwm.env.detectors import Detector
from nxwm.serve.server import Session, WorldModelServer
from nxwm.serve.transports.zmq import _parse_init_payload

ACTION_SIZE = 26 * 4  # 104 bytes — matches ZMQ ACTION_SIZE
INIT_OPCODE = 0x01  # matches ZMQ MODE_INIT


def build_app(
    server: WorldModelServer,
    *,
    detector_factory: Callable[[], Detector | None] | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to ``server``.

    ``detector_factory`` (if provided) is invoked once per WebSocket
    connection so each session gets its own detector instance with an
    isolated signal buffer. Pass ``None`` for sessions without detector
    telemetry — the UI's CV slider panel just doesn't render.
    """
    app = FastAPI(title="nxwm-serve")

    # Cross-session torch lock — single GPU, no interleaving of forward passes.
    # WebSocket handlers `await` it before calling into `Session.step`.
    torch_lock = asyncio.Lock()
    app.state.server = server
    app.state.torch_lock = torch_lock

    @app.get("/v1/info")
    async def get_info() -> dict[str, Any]:
        return asdict(server.info())

    @app.post("/v1/reload")
    async def post_reload(body: dict[str, Any]) -> dict[str, bool]:
        model_path = body.get("model_path")
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path is required")
        async with torch_lock:
            await asyncio.to_thread(server.reload, model_path)
        return {"ok": True}

    @app.get("/v1/episodes")
    async def get_episodes() -> dict[str, list[str]]:
        return {"episodes": server._list_episodes()}

    @app.exception_handler(HTTPException)
    async def http_exc_handler(_request, exc: HTTPException) -> JSONResponse:  # type: ignore[override]
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.websocket("/v1/session")
    async def session_ws(ws: WebSocket) -> None:
        await ws.accept()
        session: Session = server.new_session(
            detector=detector_factory() if detector_factory is not None else None,
        )
        # Mirror the auto-seed behavior of the default session so a client
        # that just opens the socket and starts stepping isn't dead-ended.
        if server.data_path is not None and session.state is None:
            try:
                await asyncio.to_thread(session._auto_seed_from_data_path)
            except Exception:
                # Non-fatal — the client can still send MODE_INIT or reseed.
                pass

        await ws.send_text(
            json.dumps(
                {
                    "type": "ready",
                    "history_length": server.history_length,
                    "goal_offset": server.goal_offset,
                    "action_dim": server.model.action_dims,
                    "current_episode_file": session.current_episode_file,
                }
            )
        )

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if (data := msg.get("bytes")) is not None:
                    await _handle_binary(ws, session, torch_lock, data)
                elif (text := msg.get("text")) is not None:
                    await _handle_text(ws, session, torch_lock, text)
        except WebSocketDisconnect:
            pass
        finally:
            # Drop our strong ref; WeakSet on the server lets it GC.
            del session

    return app


async def _handle_binary(
    ws: WebSocket,
    session: Session,
    torch_lock: asyncio.Lock,
    data: bytes,
) -> None:
    import numpy as np

    if len(data) == ACTION_SIZE:
        action = np.frombuffer(data, dtype=np.float32).copy()
        try:
            async with torch_lock:
                jpeg, telemetry = await asyncio.to_thread(
                    session.step_with_telemetry, action
                )
        except Exception as e:
            await ws.send_text(json.dumps({"error": f"{e}"}))
            return
        # Telemetry first (text), then JPEG (binary). Keeping them in this
        # order lets a naive client `recv_text(); recv_bytes()` without any
        # framing logic.
        await ws.send_text(json.dumps({"type": "telemetry", "telemetry": telemetry}))
        await ws.send_bytes(jpeg)
        return

    if len(data) > 0 and data[0] == INIT_OPCODE:
        try:
            frames, goal = _parse_init_payload(data[1:])
        except ValueError as e:
            await ws.send_text(json.dumps({"error": str(e)}))
            return
        try:
            async with torch_lock:
                await asyncio.to_thread(session.init_from_frames, frames, goal)
        except Exception as e:
            await ws.send_text(json.dumps({"error": f"{e}"}))
            return
        await ws.send_text(json.dumps({"type": "ok"}))
        return

    await ws.send_text(json.dumps({"error": f"unknown binary frame (len={len(data)})"}))


async def _handle_text(
    ws: WebSocket,
    session: Session,
    torch_lock: asyncio.Lock,
    text: str,
) -> None:
    try:
        body = json.loads(text)
    except json.JSONDecodeError as e:
        await ws.send_text(json.dumps({"error": f"invalid JSON: {e}"}))
        return

    op = body.get("op")
    try:
        if op == "reseed":
            file = body["file"]
            start_frame = int(body.get("start_frame", 100))
            async with torch_lock:
                await asyncio.to_thread(session.reseed, file, start_frame)
            await ws.send_text(json.dumps({"type": "ok"}))
        elif op == "detector_config":
            await ws.send_text(
                json.dumps({"type": "detector_config", "config": session.detector_config()})
            )
        elif op == "detector_params":
            params = body.get("params") or {}
            out = await asyncio.to_thread(session.apply_detector_params, params)
            await ws.send_text(json.dumps({"type": "detector_applied", **out}))
        elif op == "detector_reset":
            await asyncio.to_thread(session.reset_detector)
            await ws.send_text(json.dumps({"type": "ok"}))
        elif op == "info":
            await ws.send_text(
                json.dumps({"type": "info", "info": asdict(session._server.info())})
            )
        else:
            await ws.send_text(json.dumps({"error": f"unknown op: {op!r}"}))
    except FileNotFoundError as e:
        await ws.send_text(json.dumps({"error": f"{e}"}))
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"{e}"}))


class HTTPTransport:
    """Run a uvicorn server hosting :func:`build_app` against ``server``.

    Mirrors :class:`nxwm.serve.transports.zmq.ZMQTransport`'s shape so the
    CLI can branch on ``--transport`` without special-casing.
    """

    def __init__(
        self,
        server: WorldModelServer,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        ssl_certfile: str | None = None,
        ssl_keyfile: str | None = None,
        root_path: str = "",
        detector_factory: Callable[[], Detector | None] | None = None,
    ) -> None:
        import uvicorn

        self.server = server
        self.host = host
        self.port = port
        self.app = build_app(server, detector_factory=detector_factory)
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            root_path=root_path,
            # `websockets-sansio` avoids uvicorn's `websockets.legacy` import,
            # which emits a DeprecationWarning that pytest's filter raises.
            ws="websockets-sansio",
            ws_max_size=64 * 1024 * 1024,  # 64 MiB ceiling for JPEG batches
        )
        self._uvicorn = uvicorn.Server(config)

    def run(self) -> None:
        self._uvicorn.run()

    def stop(self) -> None:
        self._uvicorn.should_exit = True

    def shutdown(self) -> None:
        self.stop()
