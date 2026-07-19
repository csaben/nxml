from __future__ import annotations

import secrets
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from nxml_edge.supervisor import EdgeSupervisor


def create_app(supervisor: EdgeSupervisor, *, token: str) -> FastAPI:
    app = FastAPI(title="nxml-edge", version="0.1.0")

    def require_token(request: Request) -> None:
        supplied = request.headers.get("x-nxml-edge-token") or request.query_params.get("token")
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="bad or missing edge token")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = files("nxml_edge").joinpath("static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/status")
    def status(request: Request):
        require_token(request)
        return supervisor.status()

    @app.post("/api/session/start")
    def start(request: Request):
        require_token(request)
        try:
            return supervisor.start_session()
        except (OSError, RuntimeError, PermissionError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/session/retry")
    def retry(request: Request):
        require_token(request)
        return supervisor.retry()

    @app.post("/api/session/stop")
    def stop(request: Request):
        require_token(request)
        return supervisor.stop_session()

    @app.post("/api/session/eject")
    def eject(request: Request):
        require_token(request)
        return supervisor.eject()

    @app.post("/api/session/rearm")
    def rearm(request: Request):
        require_token(request)
        return supervisor.rearm()

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    return app
