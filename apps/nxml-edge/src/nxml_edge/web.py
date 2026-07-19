from __future__ import annotations

import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from nxml_edge.cluster import ClusterDashboard, ClusterError
from nxml_edge.human_control import HumanActionRequest, HumanControlBridge, HumanControlError
from nxml_edge.preview import NxbtStateClient, PreviewSource
from nxml_edge.supervisor import EdgeSupervisor


def create_app(
    supervisor: EdgeSupervisor,
    *,
    token: str | None = None,
    auth: Callable[[Request], None] | None = None,
    cluster: ClusterDashboard | None = None,
    preview: PreviewSource | None = None,
    controller: NxbtStateClient | None = None,
    human_control: HumanControlBridge | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if human_control is not None:
                human_control.close()

    app = FastAPI(title="nxml-edge", version="0.1.0", lifespan=lifespan)

    def require_token(request: Request) -> None:
        if auth is not None:
            auth(request)
            return
        if token is None:
            raise HTTPException(status_code=503, detail="edge authentication is not configured")
        supplied = request.headers.get("x-nxml-edge-token")
        authorization = request.headers.get("authorization", "")
        scheme, _, credential = authorization.partition(" ")
        if not supplied and scheme.lower() == "bearer":
            supplied = credential.strip()
        # Kept for API compatibility. The browser UI deliberately never puts
        # credentials in URLs or persistent storage.
        supplied = supplied or request.query_params.get("token")
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="bad or missing edge token")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        require_token(request)
        html = files("nxml_edge").joinpath("static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/status")
    def status(request: Request):
        require_token(request)
        return supervisor.status()

    @app.get("/api/preview.mjpeg")
    def capture_preview(request: Request) -> StreamingResponse:
        require_token(request)
        if preview is None:
            raise HTTPException(503, "capture preview is not configured")
        return StreamingResponse(
            preview.frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, private"},
        )

    @app.get("/api/controller/state")
    def controller_state(request: Request):
        require_token(request)
        if controller is None:
            return {"reachable": False, "error": "controller state is not configured"}
        return controller.status()

    @app.get("/api/preview/status")
    def preview_status(request: Request):
        require_token(request)
        if preview is None:
            return {"ok": False, "error": "capture preview is not configured"}
        return preview.status()

    def require_human_control(request: Request) -> HumanControlBridge:
        require_token(request)
        if human_control is None:
            raise HTTPException(503, "human control is not configured")
        return human_control

    @app.get("/api/human/status")
    def human_status(request: Request):
        return require_human_control(request).status()

    @app.post("/api/human/enable")
    def human_enable(request: Request):
        try:
            return require_human_control(request).enable()
        except RuntimeError as error:
            raise HTTPException(502, str(error)) from error

    @app.post("/api/human/action")
    def human_action(payload: HumanActionRequest, request: Request):
        try:
            return require_human_control(request).apply(payload)
        except HumanControlError as error:
            raise HTTPException(error.status_code, str(error)) from error

    @app.post("/api/human/disable")
    def human_disable(request: Request):
        return require_human_control(request).disable("client_disabled")

    @app.get("/api/cluster/status")
    def cluster_status(request: Request):
        require_token(request)
        return (
            cluster.status() if cluster else {"connected": False, "error": "cluster not configured"}
        )

    def cluster_action(request: Request) -> ClusterDashboard:
        require_token(request)
        if cluster is None:
            raise HTTPException(503, "cluster not configured")
        return cluster

    @app.post("/api/cluster/datasets/{dataset_id}/human-snapshot")
    def snapshot(dataset_id: str, request: Request):
        try:
            return cluster_action(request).create_snapshot(
                dataset_id, request.headers.get("idempotency-key")
            )
        except ClusterError as error:
            raise HTTPException(error.status or 503, str(error)) from error

    @app.post("/api/cluster/training/bc")
    async def train(request: Request):
        dashboard = cluster_action(request)
        body = await request.json()
        try:
            return dashboard.submit_bc(
                body["snapshot_id"], body.get("config", {}), request.headers.get("idempotency-key")
            )
        except ClusterError as error:
            raise HTTPException(error.status or 503, str(error)) from error

    @app.post("/api/cluster/models/{revision_id}/promote")
    async def promote(revision_id: str, request: Request):
        dashboard = cluster_action(request)
        body = await request.json()
        try:
            return dashboard.promote(
                revision_id,
                int(body["expected_generation"]),
                request.headers.get("idempotency-key"),
            )
        except ClusterError as error:
            raise HTTPException(error.status or 503, str(error)) from error

    @app.post("/api/cluster/deployment/rollback")
    async def rollback(request: Request):
        dashboard = cluster_action(request)
        body = await request.json()
        try:
            return dashboard.rollback(
                int(body["expected_generation"]), request.headers.get("idempotency-key")
            )
        except ClusterError as error:
            raise HTTPException(error.status or 503, str(error)) from error

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
