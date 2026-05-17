"""Route handlers for the nxwm UI.

The UI consumes a uniform info schema regardless of which client returned it.
``InProcessClient`` returns full keys (``current_model_path``,
``available_episodes``, ``architecture``, ``config``…) while
``RemoteZMQClient`` returns the wire-protocol keys (``current_model``,
``npz_files``, no architecture/config). :func:`_normalize_info` merges them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from nx_packets import ACTION_DIM

from nxwm.serve import WorldModelClient

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StepRequest(BaseModel):
    action: list[float] = Field(..., min_length=ACTION_DIM, max_length=ACTION_DIM)


class ReseedRequest(BaseModel):
    file: str
    start_frame: int = 100


class ReloadRequest(BaseModel):
    model_path: str


class DetectorParamsRequest(BaseModel):
    params: dict[str, Any]


def _normalize_info(info: dict[str, Any]) -> dict[str, Any]:
    """Translate either flavor of info() output into the UI's canonical shape."""
    return {
        "current_model": info.get("current_model_path") or info.get("current_model"),
        "episodes": info.get("available_episodes") or info.get("npz_files") or [],
        "checkpoints": info.get("available_checkpoints") or info.get("checkpoints") or [],
        "history_length": info.get("history_length"),
        "goal_offset": info.get("goal_offset"),
        "flow_steps": info.get("flow_steps"),
        "cfg_scale": info.get("cfg_scale"),
        "latent_scale": info.get("latent_scale"),
        "current_episode_frame": info.get("current_episode_frame"),
        "current_episode_file": info.get("current_episode_file"),
        "architecture": info.get("architecture"),
        "config": info.get("config"),
    }


def register_routes(app: FastAPI, *, client: WorldModelClient, game: str) -> None:
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = (STATIC_DIR / "index.html").read_text()
        return html.replace("{{GAME}}", game)

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return _normalize_info(client.info())

    @app.post("/api/step")
    def step(req: StepRequest) -> Response:
        action = np.asarray(req.action, dtype=np.float32)
        try:
            jpeg, telemetry = client.step_with_telemetry(action)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        headers: dict[str, str] = {}
        if telemetry:
            # Header is JSON-encoded so the JS side can parse it without a
            # multipart envelope. Browsers strip Set-Cookie etc but custom
            # X- headers are passed through to fetch().
            headers["X-Detector-Telemetry"] = json.dumps(telemetry)
            headers["Access-Control-Expose-Headers"] = "X-Detector-Telemetry"
        return Response(content=jpeg, media_type="image/jpeg", headers=headers)

    @app.post("/api/reseed")
    def reseed(req: ReseedRequest) -> dict[str, bool]:
        try:
            client.reseed(req.file, req.start_frame)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True}

    @app.post("/api/reload")
    def reload(req: ReloadRequest) -> dict[str, bool]:
        try:
            client.reload(req.model_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True}

    # ---- detector ----

    @app.get("/api/detector/config")
    def detector_config() -> dict[str, Any]:
        return client.detector_config()

    @app.post("/api/detector/config")
    def detector_apply(req: DetectorParamsRequest) -> dict[str, Any]:
        try:
            return client.apply_detector_params(req.params)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/api/detector/reset")
    def detector_reset() -> dict[str, bool]:
        try:
            client.reset_detector()
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True}

    @app.get("/api/detector/debug.png")
    def detector_debug() -> Response:
        img = client.detector_debug_image()
        if img is None:
            raise HTTPException(status_code=404, detail="no debug image available")
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(status_code=500, detail="png encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")
