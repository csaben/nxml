"""Browser teleop UI for ``nxml-autopilot``.

Mirrors ``nxml-collect.ui`` but routes the gamepad action into the
:class:`WebGamepadReader` (so the mux can merge it with the AI source)
instead of POSTing straight to the orchestrator. MJPEG comes off the same
``V4L2Source`` the AI inference loop already holds.

Token gate: if a non-empty token is set, every request to ``/action`` and
``/mjpeg`` must carry it. The browser picks it up from the URL query
``?token=…`` on first load and forwards it as ``X-Autopilot-Token``.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from collections.abc import AsyncIterator
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from nx_macros import MacroPlayer, MacroRecorder, MacroStore, sanitize_name
from nxml_capture.source import CaptureSource
from nxml_mux.input_devices.readers import WebGamepadReader

from nxml_autopilot.recording import RecordingController, fresh_episode_path
from nxml_autopilot.triggers import TriggerStore, TriggerWatcher

_MJPEG_BOUNDARY = "frame"
_JPEG_QUALITY = 70
_MJPEG_INTERVAL = 1 / 60


def _check_token(request: Request, expected: str) -> None:
    if not expected:
        return
    supplied = request.headers.get("x-autopilot-token") or request.query_params.get("token")
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="bad or missing token")


def create_app(reader: WebGamepadReader, source: CaptureSource, *, token: str = "") -> FastAPI:
    app = FastAPI(title="nxml-autopilot teleop", version="0.1.0")
    app.state.recorder = None
    app.state.record_root = None
    app.state.macro_recorder = None
    app.state.macro_store = None
    app.state.macro_player = None
    app.state.macro_game = None
    app.state.macro_loop = False
    app.state.trigger_store = None
    app.state.trigger_watcher = None
    app.state.runtime = None  # AutopilotRunner; set via attach_runtime

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = files("nxml_autopilot").joinpath("static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/mjpeg")
    async def mjpeg(request: Request) -> StreamingResponse:
        _check_token(request, token)

        async def stream() -> AsyncIterator[bytes]:
            while True:
                frame = source.latest()
                if frame is None:
                    await asyncio.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(
                    ".jpg",
                    frame.image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
                )
                if not ok:
                    await asyncio.sleep(0.01)
                    continue
                yield (
                    b"--"
                    + _MJPEG_BOUNDARY.encode()
                    + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                await asyncio.sleep(_MJPEG_INTERVAL)

        return StreamingResponse(
            stream(),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
        )

    @app.post("/action")
    async def post_action(request: Request) -> JSONResponse:
        _check_token(request, token)
        payload = await request.json()
        vector = payload.get("vector")
        if not isinstance(vector, list):
            raise HTTPException(status_code=400, detail="missing 'vector': number[26]")
        try:
            arr = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"vector parse: {e}") from e
        try:
            reader.push_action(arr)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse({"ok": True})

    @app.post("/recording/start")
    async def recording_start(request: Request) -> JSONResponse:
        _check_token(request, token)
        ctl: RecordingController | None = app.state.recorder
        root: Path | None = app.state.record_root
        if ctl is None or root is None:
            raise HTTPException(
                status_code=400,
                detail="recording not configured — pass --record DIR at startup",
            )
        target = fresh_episode_path(root)
        try:
            path = ctl.start(target)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return JSONResponse({"ok": True, "path": str(path)})

    @app.post("/recording/stop")
    async def recording_stop(request: Request) -> JSONResponse:
        _check_token(request, token)
        ctl: RecordingController | None = app.state.recorder
        if ctl is None:
            raise HTTPException(status_code=400, detail="recording not configured")
        out = ctl.stop()
        return JSONResponse({"ok": True, "finalized": str(out) if out else None})

    @app.get("/recording/status")
    async def recording_status(request: Request) -> JSONResponse:
        _check_token(request, token)
        ctl: RecordingController | None = app.state.recorder
        if ctl is None:
            return JSONResponse({"configured": False, "active": False})
        return JSONResponse({"configured": True, **ctl.status()})

    # ------------------------------------------------------------------
    # Macros
    # ------------------------------------------------------------------

    def _macros_status_payload() -> dict[str, Any]:
        rec: MacroRecorder | None = app.state.macro_recorder
        store: MacroStore | None = app.state.macro_store
        player: MacroPlayer | None = app.state.macro_player
        return {
            "configured": store is not None,
            "store_root": str(store.root) if store is not None else None,
            "recording": rec.is_active if rec is not None else False,
            "recording_name": rec.name if rec is not None else None,
            "recording_frames": rec.frame_count if rec is not None else 0,
            "playing": player.is_playing if player is not None else False,
            "loop": bool(app.state.macro_loop),
            "macros": store.list() if store is not None else [],
        }

    @app.get("/macros/status")
    async def macros_status(request: Request) -> JSONResponse:
        _check_token(request, token)
        return JSONResponse(_macros_status_payload())

    @app.post("/macros/record/start")
    async def macros_record_start(request: Request) -> JSONResponse:
        _check_token(request, token)
        rec: MacroRecorder | None = app.state.macro_recorder
        if rec is None:
            raise HTTPException(status_code=400, detail="macros not configured")
        payload = await request.json()
        name = payload.get("name")
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="missing 'name'")
        try:
            sanitize_name(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            rec.start(
                name,
                metadata={"game": app.state.macro_game} if app.state.macro_game else None,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return JSONResponse(_macros_status_payload())

    @app.post("/macros/record/stop")
    async def macros_record_stop(request: Request) -> JSONResponse:
        _check_token(request, token)
        rec: MacroRecorder | None = app.state.macro_recorder
        store: MacroStore | None = app.state.macro_store
        if rec is None or store is None:
            raise HTTPException(status_code=400, detail="macros not configured")
        if not rec.is_active:
            raise HTTPException(status_code=409, detail="not recording")
        macro = rec.stop()
        if not macro.frames:
            raise HTTPException(status_code=409, detail="empty macro — discarded")
        path = store.save(macro)
        return JSONResponse(
            {"ok": True, "saved": str(path), **_macros_status_payload()}
        )

    @app.post("/macros/record/cancel")
    async def macros_record_cancel(request: Request) -> JSONResponse:
        _check_token(request, token)
        rec: MacroRecorder | None = app.state.macro_recorder
        if rec is None:
            raise HTTPException(status_code=400, detail="macros not configured")
        rec.cancel()
        return JSONResponse(_macros_status_payload())

    @app.post("/macros/play")
    async def macros_play(request: Request) -> JSONResponse:
        _check_token(request, token)
        store: MacroStore | None = app.state.macro_store
        player: MacroPlayer | None = app.state.macro_player
        if store is None or player is None:
            raise HTTPException(status_code=400, detail="macros not configured")
        payload = await request.json()
        name = payload.get("name")
        loop = bool(payload.get("loop", False))
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="missing 'name'")
        if player.is_playing:
            raise HTTPException(status_code=409, detail="playback already in progress")
        try:
            macro = store.load(name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        app.state.macro_loop = loop
        try:
            player.play_async(macro, loop=loop)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return JSONResponse(_macros_status_payload())

    @app.post("/macros/stop")
    async def macros_stop(request: Request) -> JSONResponse:
        _check_token(request, token)
        player: MacroPlayer | None = app.state.macro_player
        if player is None:
            raise HTTPException(status_code=400, detail="macros not configured")
        player.stop()
        app.state.macro_loop = False
        return JSONResponse(_macros_status_payload())

    @app.post("/macros/delete")
    async def macros_delete(request: Request) -> JSONResponse:
        _check_token(request, token)
        store: MacroStore | None = app.state.macro_store
        if store is None:
            raise HTTPException(status_code=400, detail="macros not configured")
        payload = await request.json()
        name = payload.get("name")
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="missing 'name'")
        try:
            sanitize_name(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        store.delete(name)
        return JSONResponse(_macros_status_payload())

    # ------------------------------------------------------------------
    # Triggers (visual-detect → run macro)
    # ------------------------------------------------------------------

    def _triggers_status_payload() -> dict[str, Any]:
        store: TriggerStore | None = app.state.trigger_store
        watcher: TriggerWatcher | None = app.state.trigger_watcher
        if store is None or watcher is None:
            return {
                "configured": False,
                "store_root": None,
                "triggers": [],
                "armed": [],
                "states": {},
            }
        return {
            "configured": True,
            "store_root": str(store.root),
            "triggers": store.list(),
            **watcher.status(),
        }

    def _require_triggers() -> tuple[TriggerStore, TriggerWatcher]:
        store: TriggerStore | None = app.state.trigger_store
        watcher: TriggerWatcher | None = app.state.trigger_watcher
        if store is None or watcher is None:
            raise HTTPException(
                status_code=400,
                detail="triggers not configured — pass --trigger-dir DIR at startup",
            )
        return store, watcher

    @app.get("/triggers/status")
    async def triggers_status(request: Request) -> JSONResponse:
        _check_token(request, token)
        return JSONResponse(_triggers_status_payload())

    @app.get("/triggers/get/{name}")
    async def triggers_get(request: Request, name: str) -> JSONResponse:
        _check_token(request, token)
        store, _ = _require_triggers()
        try:
            spec = store.load(name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return JSONResponse(spec.model_dump())

    @app.post("/triggers/arm")
    async def triggers_arm(request: Request) -> JSONResponse:
        _check_token(request, token)
        _, watcher = _require_triggers()
        payload = await request.json()
        name = payload.get("name")
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="missing 'name'")
        try:
            watcher.arm(name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse(_triggers_status_payload())

    @app.post("/triggers/disarm")
    async def triggers_disarm(request: Request) -> JSONResponse:
        _check_token(request, token)
        _, watcher = _require_triggers()
        payload = await request.json()
        name = payload.get("name")
        if name is None:
            watcher.disarm_all()
        elif isinstance(name, str):
            watcher.disarm(name)
        else:
            raise HTTPException(status_code=400, detail="'name' must be a string")
        return JSONResponse(_triggers_status_payload())

    # Fields the UI is allowed to edit live. Anything outside this set is
    # rejected to keep the JSON schema honest (image_filename, name,
    # detector_kind, etc. should be set at seed time, not poked from a
    # number input).
    _TRIGGER_LIVE_FIELDS = {
        "cooldown_sec",
        "debounce_sec",
        "similarity_threshold",
        "mash_duration_sec",
        "mash_frames_per_phase",
    }

    @app.post("/triggers/update")
    async def triggers_update(request: Request) -> JSONResponse:
        _check_token(request, token)
        store, watcher = _require_triggers()
        payload = await request.json()
        name = payload.get("name")
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="missing 'name'")
        try:
            spec = store.load(name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        updates = {k: v for k, v in payload.items() if k != "name"}
        bad = set(updates) - _TRIGGER_LIVE_FIELDS
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"cannot edit fields {sorted(bad)} live; allowed: "
                f"{sorted(_TRIGGER_LIVE_FIELDS)}",
            )
        patched = spec.model_dump()
        patched.update(updates)
        try:
            updated = type(spec).model_validate(patched)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"validation: {e}") from e
        store.save(updated)
        watcher.update_armed_spec(name, **updates)
        return JSONResponse({"ok": True, "spec": updated.model_dump()})

    @app.get("/triggers/images/{filename}")
    async def triggers_images_get(request: Request, filename: str) -> FileResponse:
        _check_token(request, token)
        store, _ = _require_triggers()
        # Reject path traversal; the directory is flat by design.
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="bad filename")
        path = store.images_dir / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    # ------------------------------------------------------------------
    # Runtime — AI overlay toggle + mux mode switch
    # ------------------------------------------------------------------

    def _runtime_or_400():
        rt = app.state.runtime
        if rt is None:
            raise HTTPException(status_code=400, detail="runtime not attached")
        return rt

    @app.get("/runtime/status")
    async def runtime_status(request: Request) -> JSONResponse:
        _check_token(request, token)
        rt = app.state.runtime
        if rt is None:
            return JSONResponse({"attached": False})
        return JSONResponse({"attached": True, **rt.runtime_status()})

    @app.post("/runtime/ai")
    async def runtime_ai(request: Request) -> JSONResponse:
        _check_token(request, token)
        rt = _runtime_or_400()
        payload = await request.json()
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="missing 'enabled': bool")
        rt.set_ai_enabled(enabled)
        return JSONResponse({"attached": True, **rt.runtime_status()})

    @app.post("/runtime/mode")
    async def runtime_mode(request: Request) -> JSONResponse:
        _check_token(request, token)
        rt = _runtime_or_400()
        payload = await request.json()
        mode = payload.get("mode")
        if not isinstance(mode, str):
            raise HTTPException(status_code=400, detail="missing 'mode': str")
        try:
            rt.set_mode(mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse({"attached": True, **rt.runtime_status()})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "token_required": bool(token)}

    return app


class AutopilotWebServer:
    """uvicorn server bound to a daemon thread."""

    def __init__(
        self,
        reader: WebGamepadReader,
        source: CaptureSource,
        *,
        host: str,
        port: int,
        token: str = "",
    ) -> None:
        self.app = create_app(reader, source, token=token)
        self._config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(self._config)
        self._thread: threading.Thread | None = None
        self.host = host
        self.port = port
        self.token = token

    def attach_recorder(self, controller: RecordingController, *, root: Path | None) -> None:
        self.app.state.recorder = controller
        self.app.state.record_root = root

    def attach_macros(
        self,
        *,
        recorder: MacroRecorder,
        store: MacroStore | None,
        player: MacroPlayer,
        game: str | None,
    ) -> None:
        self.app.state.macro_recorder = recorder
        self.app.state.macro_store = store
        self.app.state.macro_player = player
        self.app.state.macro_game = game

    def attach_triggers(
        self,
        *,
        store: TriggerStore | None,
        watcher: TriggerWatcher | None,
    ) -> None:
        """Hand the trigger store + watcher to the web app."""
        self.app.state.trigger_store = store
        self.app.state.trigger_watcher = watcher

    def attach_runtime(self, runtime: Any) -> None:
        """Hand a runtime handle to the web app for ``/runtime/*`` endpoints.

        ``runtime`` must expose ``runtime_status() -> dict``,
        ``set_ai_enabled(bool)``, and ``set_mode(str)``. The concrete type is
        ``AutopilotRunner`` but kept ``Any`` here to avoid the import cycle.
        """
        self.app.state.runtime = runtime

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name="nxml-autopilot-web",
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def wait_ready(self, timeout: float = 5.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return True
            time.sleep(0.05)
        return False
