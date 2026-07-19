#!/usr/bin/env python3
"""Fixture-backed end-to-end smoke for the cradle-ns operator health contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        payloads = {
            "/orchestrator/health": {"running": True, "connected": True},
            "/autopilot/health": {"ok": True, "token_required": True},
            "/autopilot/runtime/status": {
                "attached": True,
                "mode": "human-takeover",
                "ai_enabled": False,
            },
        }
        payload = payloads.get(self.path)
        if payload is None:
            self.send_error(404)
            return
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        return


def load_health_module():
    path = Path(__file__).with_name("nxml_edge_health.py")
    spec = importlib.util.spec_from_file_location("nxml_edge_health", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inject", choices=["none", "autopilot-down", "stale-spool"], default="none"
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "captures"
            state = root / "spool"
            capture.mkdir()
            state.mkdir()
            updated_at = time.time() - (3600 if args.inject == "stale-spool" else 0)
            (state / "status.json").write_text(json.dumps({"updated_at": updated_at}))
            port = server.server_address[1]
            autopilot_port = port + 1 if args.inject == "autopilot-down" else port
            env = {
                "NXML_ORCHESTRATOR_URL": f"http://127.0.0.1:{port}/orchestrator",
                "NXML_AUTOPILOT_URL": f"http://127.0.0.1:{autopilot_port}/autopilot",
                "NXML_CAPTURE_DEVICE": "/dev/null",
                "NXML_CAPTURE_REQUIRE_DURABLE_ACCESS": "0",
                "NXML_CAPTURE_DIR": str(capture),
                "NXML_SPOOL_STATE_DIR": str(state),
                "NXML_SPOOL_STATUS_MAX_AGE_SECONDS": "60",
                "NXML_DISK_MIN_FREE_GB": "0",
                "AUTOPILOT_WEB_TOKEN": "fixture-token",
            }
            report = load_health_module().build_report(env)
            print(json.dumps(report, indent=2, sort_keys=True))
            expected_ok = args.inject == "none"
            return 0 if report["ok"] == expected_ok else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
