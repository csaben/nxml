"""Live training viewer: a file-based FastAPI app over a run directory.

Reads only what the trainer writes (status.json / metrics.jsonl / recons/*.gif) — no GPU,
no torch, works mid-run, post-run, or over a network mount. Modeled on nxwm's ui/app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def build_app(run_dir: str | Path) -> FastAPI:
    run_dir = Path(run_dir)
    app = FastAPI(title=f"nxwm-mira watch: {run_dir.name}")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text().replace("{{RUN_DIR}}", str(run_dir))

    @app.get("/api/status")
    def status() -> JSONResponse:
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            raise HTTPException(404, f"No status.json in {run_dir} yet — is training running?")
        payload = json.loads(status_path.read_text())
        recons_dir = run_dir / "recons"
        payload["recons"] = (
            sorted(p.name for p in recons_dir.glob("step_*.gif")) if recons_dir.is_dir() else []
        )
        return JSONResponse(payload)

    @app.get("/api/history")
    def history(n: int = 500) -> JSONResponse:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            return JSONResponse([])
        lines = metrics_path.read_text().splitlines()[-n:]
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line mid-write
        return JSONResponse(records)

    recons_dir = run_dir / "recons"
    recons_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/recons", StaticFiles(directory=recons_dir), name="recons")

    return app
