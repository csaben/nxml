from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
recording_mod = load("dagger_recording", DEPLOY / "dagger_recording.py")
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
        capture_dir=tmp_path,
        request_timeout=0.01,
    )
    wire = reader.refresh().wire()
    assert wire["schema_version"] == "nxml.dagger-operations-status.v1"
    assert wire["session"]["recording_available"] is False
    assert wire["spool"]["commit_id"] == "receipt-7"
    assert wire["spool"]["local_buffered_bytes"] >= spool.stat().st_size
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


def test_recording_controls_are_explicit_and_do_not_add_auth():
    class Recorder:
        state = "idle"

        def status(self):
            return {"state": self.state, "frames": 0, "duration_seconds": 0}

        def start(self):
            self.state = "recording"
            return self.status()

        def stop(self):
            self.state = "finalized"
            return self.status()

    recorder = Recorder()
    app = minui.create_app(Orchestrator(), "/dev/null", recorder=recorder)
    with TestClient(app) as client:
        assert client.post("/api/recording/start").json()["state"] == "recording"
        assert client.post("/api/recording/stop").json()["state"] == "finalized"


def test_disk_pressure_closes_recording_admission():
    class Status:
        def __init__(self):
            self.spool = {"admission_open": False, "blocked_reason": "disk pressure"}
        def wire(self): return {}
    class Operations:
        def start(self): pass
        def stop(self): pass
        def snapshot(self): return Status()
    class Recorder:
        def status(self): return {"state": "idle"}
        def start(self): raise AssertionError("recorder must not start")

    app = minui.create_app(Orchestrator(), "/dev/null", Operations(), recorder=Recorder())
    with TestClient(app) as client:
        response = client.post("/api/recording/start")
    assert response.status_code == 409
    assert response.json()["detail"] == "disk pressure"


def test_recording_session_finalizes_and_stamps_integrity(monkeypatch, tmp_path):
    class Controller:
        def __init__(self, **_kwargs): pass
        def start(self): pass
        def stop(self): pass

    class Sync:
        invalid_samples = 0
        def __init__(self, *_args, **_kwargs): pass
        def frames(self):
            yield SimpleNamespace(valid=True)

    class Writer:
        episode_name = "episode"
        def __init__(self):
            self.count = 0
            self.config = {}
        def append(self, _synced): self.count += 1
        def __len__(self): return self.count
        def close(self): return None

    monkeypatch.setattr(recording_mod, "ControllerSubscription", Controller)
    monkeypatch.setattr(recording_mod, "Synchronizer", Sync)
    session = recording_mod.HumanRecordingSession(object(), output_dir=tmp_path)
    writer = Writer()
    session._record(writer)
    assert session.status()["state"] == "finalized"
    assert session.status()["frames"] == 1
    assert writer.config["capture_integrity"] == {"status": "complete", "error": None}


def test_recording_defaults_to_canonical_ffv1_mkv(tmp_path):
    session = recording_mod.HumanRecordingSession(object(), output_dir=tmp_path)
    assert session.codec == "ffv1"


def test_recording_loss_is_a_visible_failed_state(monkeypatch, tmp_path):
    class Controller:
        def __init__(self, **_kwargs): pass
        def start(self): pass
        def stop(self): pass

    class Sync:
        invalid_samples = 0
        def __init__(self, *_args, **_kwargs): pass
        def frames(self): raise recording_mod.CaptureFrameLossError("lost source frames")

    class Writer:
        episode_name = "episode"
        def __init__(self):
            self.config = {}
        def append_event(self, *_args, **_kwargs): pass
        def close(self): return None

    monkeypatch.setattr(recording_mod, "ControllerSubscription", Controller)
    monkeypatch.setattr(recording_mod, "Synchronizer", Sync)
    session = recording_mod.HumanRecordingSession(object(), output_dir=tmp_path)
    writer = Writer()
    session._record(writer)
    assert session.status()["state"] == "failed"
    assert "lost source frames" in session.status()["error"]
    assert writer.config["capture_integrity"]["status"] == "failed"
