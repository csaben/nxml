"""Browser teleop UI for ``nxml-collect``.

A small FastAPI app that does three things:

  1. Serves a static page (``static/index.html``) that uses the browser
     Gamepad API to poll the connected pad and POST a 26-dim action vector
     to ``/action`` ~60 Hz.
  2. Streams the live capture via ``GET /mjpeg`` as
     ``multipart/x-mixed-replace`` so the page can show what's on screen.
  3. Forwards ``POST /action`` to ``nxbt-orchestrator``'s HTTP ``/action``
     endpoint, so the browser doesn't need CORS configured against the
     orchestrator.

The MJPEG stream pulls from ``CaptureSource.latest()``, so the recorder and
the UI share a single v4l2 handle.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from importlib.resources import files
from typing import Any
from urllib.parse import urlparse

import cv2
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from nxml_capture.source import CaptureSource

_MJPEG_BOUNDARY = "frame"
_JPEG_QUALITY = 70
_MJPEG_INTERVAL = 1 / 60  # cap stream FPS independently of camera FPS


def derive_orchestrator_http(ws_url: str) -> str:
    """``ws://host:port/ws/state`` → ``http://host:port`` (scheme normalized)."""
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    netloc = parsed.netloc or "127.0.0.1:7777"
    return f"{scheme}://{netloc}"


def create_app(source: CaptureSource, orchestrator_http_url: str) -> FastAPI:
    app = FastAPI(title="nxml-collect teleop", version="0.1.0")
    client = httpx.AsyncClient(base_url=orchestrator_http_url, timeout=5.0)
    app.state.client = client

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = files("nxml_collect").joinpath("static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/mjpeg")
    async def mjpeg() -> StreamingResponse:
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
        payload = await request.json()
        try:
            r = await client.post("/action", json=payload)
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.HTTPError as e:
            return JSONResponse({"error": str(e)}, status_code=502)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "orchestrator": orchestrator_http_url,
            "source_open": getattr(source, "is_open", None),
        }

    return app


class UiServer:
    """Wraps a uvicorn server running on a daemon thread."""

    def __init__(
        self,
        source: CaptureSource,
        *,
        host: str,
        port: int,
        orchestrator_http_url: str,
    ) -> None:
        self.app = create_app(source, orchestrator_http_url)
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

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name="nxml-collect-ui",
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
