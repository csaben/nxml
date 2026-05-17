"""FastAPI app factory for the nxwm web UI.

The app is a thin shell around a :class:`WorldModelClient`. Whether that
client is :class:`InProcessClient` or :class:`RemoteZMQClient` is invisible
to the UI — every endpoint goes through the same protocol surface.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nxwm.serve import WorldModelClient
from nxwm.ui.routes import register_routes

STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_app(*, client: WorldModelClient, game: str) -> FastAPI:
    """Build the UI FastAPI app bound to ``client``.

    The returned app exposes:
      - ``GET  /``                     — HTML shell (served from ``static/index.html``)
      - ``GET  /api/info``             — normalized info dict
      - ``POST /api/step``             — body: ``{"action": [26 floats]}`` → JPEG
      - ``POST /api/reseed``           — body: ``{"file": str, "start_frame": int}``
      - ``POST /api/reload``           — body: ``{"model_path": str}``
      - ``GET  /static/...``           — JS/CSS assets
    """
    app = FastAPI(title=f"nxwm-ui ({game})")
    register_routes(app, client=client, game=game)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
