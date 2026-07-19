from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
from fastapi.testclient import TestClient
from nxml_autopilot.runner import AutopilotRunner
from nxml_autopilot.web import create_app
from nxml_capture.source import Frame
from nxml_mux.input_devices.readers import WebGamepadReader


class FakeSource:
    is_open = True

    def __init__(self) -> None:
        self.frame = Frame(
            timestamp=time.time(),
            monotonic_ns=time.monotonic_ns(),
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )

    def latest(self):
        return self.frame


class FakeRuntime:
    def __init__(self) -> None:
        self.ejected = False
        self.eject_calls = 0

    def emergency_eject(self) -> None:
        self.ejected = True
        self.eject_calls += 1

    def rearm(self) -> None:
        self.ejected = False

    def runtime_status(self):
        return {"ejected": self.ejected, "active_driver": "safety" if self.ejected else "none"}


def test_eject_api_is_authenticated_and_idempotent() -> None:
    app = create_app(WebGamepadReader(), FakeSource(), token="secret")
    runtime = FakeRuntime()
    app.state.runtime = runtime
    client = TestClient(app)

    assert client.post("/runtime/eject").status_code == 401
    headers = {"X-Autopilot-Token": "secret"}
    assert client.post("/runtime/eject", headers=headers).json()["ejected"] is True
    assert client.post("/runtime/eject", headers=headers).json()["ejected"] is True
    assert runtime.eject_calls == 2
    assert client.post("/runtime/rearm", headers=headers).json()["ejected"] is False


def _bare_runner(state_path: Path) -> AutopilotRunner:
    runner = AutopilotRunner.__new__(AutopilotRunner)
    runner.config = SimpleNamespace(eject_state_path=state_path)
    runner._ejected = threading.Event()
    runner._driver_lock = threading.Lock()
    runner._active_driver = "none"
    runner._driver_detail = "starting"
    runner._ai_source = SimpleNamespace(enabled=True)
    return runner


def test_eject_latch_survives_process_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "eject.json"
    first = _bare_runner(state_path)
    first._ejected.set()
    first._persist_eject_state()

    restarted = _bare_runner(state_path)
    restarted._load_eject_state()
    assert restarted._ejected.is_set()
    assert restarted._ai_source.enabled is False
    assert restarted._active_driver == "safety"
    assert restarted._driver_detail == "ejected"
    assert json.loads(state_path.read_text()) == {"latched": True}


class FakeAi:
    inference_count = 4

    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp

    def latest_action(self):
        return np.zeros(26, dtype=np.float32), self.timestamp

    def health(self):
        return {"last_latency_s": 0.012, "last_error": None}


def test_capture_staleness_suppresses_policy_readiness(tmp_path: Path) -> None:
    runner = _bare_runner(tmp_path / "eject.json")
    now = time.time()
    runner.config = SimpleNamespace(
        eject_state_path=tmp_path / "eject.json",
        policy_uri="zmq+frames://gpu:5557",
        camera_id=0,
        spool_status_path=None,
    )
    runner._source = FakeSource()
    runner._source.frame = Frame(
        timestamp=now - 2,
        monotonic_ns=time.monotonic_ns() - 2_000_000_000,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    runner._human = SimpleNamespace(
        source_id="web:gamepad",
        last_update_timestamp=now - 0.1,
        last_meaningful_input_timestamp=now - 5,
    )
    runner._ai = FakeAi(now - 0.1)
    runner._policy_id = "za-ppo"
    runner._policy_revision = "rev-7"
    runner._mode = "human-takeover"
    runner._orchestrator_health = lambda _now: {"connected": True}
    runner._spool_status = lambda _now: {"available": False, "admission_open": True}

    status = runner.runtime_status()
    assert status["capture"]["stale"] is True
    assert status["capture"]["last_frame_monotonic_ns"] is not None
    assert status["policy"]["ready"] is False
    assert status["controller"]["transport_fresh"] is True
    assert status["controller"]["meaningful_input_age_ms"] > 4_000


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"connected": True, "running": True}


class RecoveringHttp:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("offline")
        return FakeResponse()


def test_orchestrator_health_counts_error_then_recovery(tmp_path: Path) -> None:
    runner = _bare_runner(tmp_path / "eject.json")
    runner.config = SimpleNamespace(controller_url="http://controller")
    runner._http = RecoveringHttp()
    runner._orchestrator_status_lock = threading.Lock()
    runner._orchestrator_status = {}
    runner._orchestrator_checked_at = 0.0
    runner._orchestrator_success_count = 0
    runner._orchestrator_error_count = 0

    failed = runner._orchestrator_health(10.0)
    recovered = runner._orchestrator_health(12.0)
    assert failed["reachable"] is False
    assert recovered["connected"] is True
    assert recovered["success_count"] == 1
    assert recovered["error_count"] == 1
