from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parents[2]
DEPLOY = ROOT / "deploy" / "cradle-ns"
sys.path.insert(0, str(DEPLOY))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


status_mod = load("dagger_status", DEPLOY / "dagger_status.py")
minui = load("dagger_minui", DEPLOY / "minui.py")


class Orchestrator:
    def __init__(self):
        self.actions = []

    def post_action(self, vector):
        self.actions.append(vector)


def test_status_models_preserve_contract_and_spool_metadata(tmp_path):
    spool = tmp_path / "status.json"
    spool.write_text(json.dumps({"pending_episodes": 3, "commit_id": "receipt-7"}))
    reader = status_mod.OperationsReader(
        spool_status_path=spool,
        cluster_url="http://127.0.0.1:1",
        cluster_token_path=None,
        request_timeout=0.01,
    )
    wire = reader.refresh().wire()
    assert wire["schema_version"] == "nxml.dagger-operations-status.v1"
    assert wire["session"]["recording_available"] is False
    assert wire["spool"]["commit_id"] == "receipt-7"
    assert "cluster" in wire["errors"]


def test_slow_operations_never_block_fast_websocket(tmp_path):
    class SlowReader:
        def start(self): pass
        def stop(self): pass
        def snapshot(self):
            time.sleep(1)

    orchestrator = Orchestrator()
    app = minui.create_app(orchestrator, "/dev/null", SlowReader())
    with TestClient(app) as client:
        started = time.monotonic()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"vector": [0.0] * 26})
        assert time.monotonic() - started < 0.5
    assert len(orchestrator.actions) >= 2  # packet plus disconnect neutral


def test_unconfigured_shell_is_no_auth_and_read_only():
    app = minui.create_app(Orchestrator(), "/dev/null")
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        status = client.get("/api/ops/status").json()
        assert status["session"]["mode"] == "human"
        assert client.post("/api/ops/status").status_code == 405


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1", "192.168.1.2", "::1", "cradle-ns"])
def test_bind_rejects_non_tailnet_interfaces(host):
    with pytest.raises(ValueError, match="Tailscale"):
        minui.validate_tailnet_bind(host)


def test_bind_accepts_cradle_ns_tailnet_ip():
    assert minui.validate_tailnet_bind("100.73.109.68") == "100.73.109.68"
