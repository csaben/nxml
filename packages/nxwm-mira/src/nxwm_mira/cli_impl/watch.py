"""`nxwm-mira watch`: serve the live training viewer for a run directory."""

from __future__ import annotations


def run_watch(run_dir: str, host: str, port: int) -> None:
    import uvicorn

    from nxwm_mira.ui.app import build_app

    print(f"Watching {run_dir} at http://{host}:{port}")
    uvicorn.run(build_app(run_dir), host=host, port=port, log_level="warning")
