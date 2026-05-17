"""UI smoke tests using FastAPI's TestClient.

We test the route surface against a ``StubClient`` rather than a real model so
the tests are fast and don't depend on a VAE. The two integration points that
matter for this layer are:

  - URL routing (``/``, ``/api/info``, ``/api/step``, ``/api/reseed``, ``/api/reload``)
  - Info-shape normalization (modern vs legacy keys → canonical UI shape)

Server-level equivalence is verified separately in test_server_core.py and
test_clients.py — the UI layer is "just plumbing" by design.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from nxwm.ui.app import build_app


class _StubClient:
    """In-memory client that records calls — satisfies WorldModelClient."""

    def __init__(self, *, info_shape: str = "modern"):
        self.calls: list[tuple] = []
        self._info_shape = info_shape
        self._reseeded = False
        # Optional detector hooks; tests opt in.
        self._detector_cfg: dict[str, Any] = {}
        self._telemetry_for_next_step: dict[str, Any] = {}

    def step(self, action: np.ndarray) -> bytes:
        if not self._reseeded:
            raise RuntimeError("not seeded; call reseed first")
        self.calls.append(("step", action.copy()))
        # ~2 KB of "JPEG-shaped" placeholder bytes — content doesn't matter for the test
        return b"\xff\xd8\xff\xe0" + b"x" * 256 + b"\xff\xd9"

    def step_with_telemetry(self, action: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        return self.step(action), self._telemetry_for_next_step

    # Detector surface — empty by default; tests opt in by setting attrs.
    def detector_config(self) -> dict[str, Any]:
        return self._detector_cfg

    def apply_detector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._detector_cfg:
            raise RuntimeError("no detector configured on this server")
        self.calls.append(("apply_detector_params", dict(params)))
        return {"state": {"detected": False, "streak": 0}, "params": params}

    def reset_detector(self) -> None:
        if not self._detector_cfg:
            raise RuntimeError("no detector configured on this server")
        self.calls.append(("reset_detector",))

    def detector_debug_image(self):
        return None

    def reseed(self, file: str, start_frame: int = 100) -> None:
        if file == "missing.npz":
            raise FileNotFoundError(f"no such episode: {file}")
        self.calls.append(("reseed", file, start_frame))
        self._reseeded = True

    def reload(self, model_path: str) -> None:
        if not model_path:
            raise ValueError("empty model_path")
        self.calls.append(("reload", model_path))

    def info(self) -> dict[str, Any]:
        if self._info_shape == "modern":
            return {
                "current_model_path": "/stub.pt",
                "architecture": "dit_v1",
                "config": {"embed_dim": 64},
                "history_length": 10,
                "goal_offset": 30,
                "flow_steps": 5,
                "cfg_scale": 1.0,
                "latent_scale": 0.18215,
                "current_episode_frame": 7,
                "current_episode_file": "ep_a.npz",
                "available_episodes": ["ep_a.npz", "ep_b.npz"],
                "available_checkpoints": ["/c/best.pt"],
            }
        # legacy wire shape (what RemoteZMQClient returns)
        return {
            "current_model": "/stub.pt",
            "npz_files": ["ep_a.npz", "ep_b.npz"],
            "checkpoints": ["/c/best.pt"],
            "flow_steps": 5,
            "cfg_scale": 1.0,
            "history_length": 10,
            "goal_offset": 30,
            "latent_scale": 0.18215,
            "current_episode_frame": 7,
        }

    def init_from_frames(self, frames_jpeg, goal_jpeg=None):
        self.calls.append(("init", len(frames_jpeg), goal_jpeg is not None))

    def close(self):
        pass


@pytest.fixture
def stub_app():
    stub = _StubClient(info_shape="modern")
    app = build_app(client=stub, game="test-game")
    return stub, TestClient(app)


@pytest.fixture
def legacy_stub_app():
    stub = _StubClient(info_shape="legacy")
    app = build_app(client=stub, game="legacy-game")
    return stub, TestClient(app)


def test_index_html_served(stub_app):
    _, tc = stub_app
    r = tc.get("/")
    assert r.status_code == 200
    assert "test-game" in r.text
    # The shell references the JS/CSS we ship — sanity check.
    assert "/static/ui.js" in r.text
    assert "/static/ui.css" in r.text


def test_static_assets_served(stub_app):
    _, tc = stub_app
    for asset in ("ui.js", "ui.css"):
        r = tc.get(f"/static/{asset}")
        assert r.status_code == 200, asset
        assert len(r.content) > 0


def test_info_modern_shape(stub_app):
    _, tc = stub_app
    r = tc.get("/api/info")
    assert r.status_code == 200
    info = r.json()
    assert info["current_model"] == "/stub.pt"
    assert info["episodes"] == ["ep_a.npz", "ep_b.npz"]
    assert info["checkpoints"] == ["/c/best.pt"]
    assert info["history_length"] == 10
    assert info["architecture"] == "dit_v1"
    assert info["config"]["embed_dim"] == 64
    assert info["current_episode_file"] == "ep_a.npz"


def test_info_legacy_wire_shape_normalizes(legacy_stub_app):
    """Legacy wire keys (npz_files, current_model) must surface in the UI's canonical shape."""
    _, tc = legacy_stub_app
    info = tc.get("/api/info").json()
    assert info["current_model"] == "/stub.pt"  # legacy key 'current_model'
    assert info["episodes"] == ["ep_a.npz", "ep_b.npz"]  # from legacy 'npz_files'
    assert info["history_length"] == 10
    # Modern-only fields fall back to None when not present in the legacy info.
    assert info["architecture"] is None
    assert info["config"] is None


def test_step_returns_jpeg_after_reseed(stub_app):
    stub, tc = stub_app
    r = tc.post("/api/reseed", json={"file": "ep_a.npz", "start_frame": 0})
    assert r.status_code == 200 and r.json() == {"ok": True}

    r = tc.post("/api/step", json={"action": [0.0] * 26})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 0
    assert any(c[0] == "step" for c in stub.calls)


def test_step_action_shape_validation(stub_app):
    _, tc = stub_app
    tc.post("/api/reseed", json={"file": "ep_a.npz", "start_frame": 0})
    # Wrong action length → 422 from pydantic.
    r = tc.post("/api/step", json={"action": [0.0] * 25})
    assert r.status_code == 422


def test_step_before_reseed_returns_400(stub_app):
    _, tc = stub_app
    r = tc.post("/api/step", json={"action": [0.0] * 26})
    assert r.status_code == 400
    assert "not seeded" in r.json()["detail"]


def test_reseed_unknown_file_returns_400(stub_app):
    _, tc = stub_app
    r = tc.post("/api/reseed", json={"file": "missing.npz", "start_frame": 0})
    assert r.status_code == 400
    assert "no such episode" in r.json()["detail"]


def test_reload_propagates_errors(stub_app):
    _, tc = stub_app
    r = tc.post("/api/reload", json={"model_path": ""})
    assert r.status_code == 400
    assert "empty model_path" in r.json()["detail"]


def test_reload_success(stub_app):
    stub, tc = stub_app
    r = tc.post("/api/reload", json={"model_path": "/new.pt"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert ("reload", "/new.pt") in stub.calls


def test_action_layout_round_trip_via_step(stub_app):
    """Buttons set in the UI must reach the client at the canonical indices.

    The JS side packs sticks to indices 0..3 and buttons to 4..25 (per
    nx_packets.action_spec). We verify the wire side does no remapping.
    """
    from nx_packets import BUTTON_INDEX

    stub, tc = stub_app
    tc.post("/api/reseed", json={"file": "ep_a.npz", "start_frame": 0})

    action = [0.0] * 26
    action[BUTTON_INDEX["A"]] = 1.0
    action[BUTTON_INDEX["DPAD_UP"]] = 1.0
    action[0] = 0.5  # L stick X
    r = tc.post("/api/step", json={"action": action})
    assert r.status_code == 200

    last = stub.calls[-1]
    assert last[0] == "step"
    sent = last[1]
    assert sent[BUTTON_INDEX["A"]] == 1.0
    assert sent[BUTTON_INDEX["DPAD_UP"]] == 1.0
    assert sent[0] == pytest.approx(0.5)


# ---------- detector ----------


def test_detector_config_empty_when_no_detector(stub_app):
    """No detector configured → /api/detector/config returns {}, step has no header."""
    stub, tc = stub_app
    assert tc.get("/api/detector/config").json() == {}
    tc.post("/api/reseed", json={"file": "ep_a.npz", "start_frame": 0})
    r = tc.post("/api/step", json={"action": [0.0] * 26})
    assert "x-detector-telemetry" not in {k.lower() for k in r.headers}


def test_detector_config_returns_schema():
    stub = _StubClient(info_shape="modern")
    stub._detector_cfg = {
        "name": "fake:det",
        "params": {"threshold": 0.2},
        "schema": {
            "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "label": "t"},
        },
        "static_meta": {"kind": "fake:det"},
    }
    tc = TestClient(build_app(client=stub, game="g"))
    cfg = tc.get("/api/detector/config").json()
    assert cfg["name"] == "fake:det"
    assert "threshold" in cfg["schema"]
    assert cfg["params"]["threshold"] == 0.2


def test_step_emits_detector_telemetry_header():
    stub = _StubClient(info_shape="modern")
    tc = TestClient(build_app(client=stub, game="g"))
    tc.post("/api/reseed", json={"file": "ep_a.npz", "start_frame": 0})

    stub._telemetry_for_next_step = {
        "name": "fake:det",
        "signals": {"score": 0.42, "sat": 100.0},
        "state": {"detected": False, "streak": 3},
        "params": {"threshold": 0.5},
    }
    r = tc.post("/api/step", json={"action": [0.0] * 26})
    assert r.status_code == 200
    raw = r.headers["x-detector-telemetry"]
    import json as _json
    tel = _json.loads(raw)
    assert tel["state"]["streak"] == 3
    assert tel["signals"]["score"] == pytest.approx(0.42)


def test_apply_detector_params_round_trip():
    stub = _StubClient(info_shape="modern")
    stub._detector_cfg = {"name": "fake:det", "params": {"threshold": 0.2}, "schema": {}}
    tc = TestClient(build_app(client=stub, game="g"))
    r = tc.post("/api/detector/config", json={"params": {"threshold": 0.6}})
    assert r.status_code == 200
    out = r.json()
    assert out["params"]["threshold"] == 0.6
    assert ("apply_detector_params", {"threshold": 0.6}) in stub.calls


def test_apply_detector_params_without_detector_returns_400(stub_app):
    _, tc = stub_app
    r = tc.post("/api/detector/config", json={"params": {"threshold": 0.6}})
    assert r.status_code == 400


def test_detector_debug_image_returns_404_when_unavailable(stub_app):
    _, tc = stub_app
    r = tc.get("/api/detector/debug.png")
    assert r.status_code == 404


def test_pokemon_za_adapter_decide_is_stateless():
    """`decide` over a recorded history must respect the current params and
    not depend on prior `decide` calls — load-bearing for offline replay."""
    import nxml_games.pokemon_za.detector_adapter  # noqa: F401
    from nxwm.env.detectors import detector_registry

    from nxml_games.pokemon_za.assets import default_end_screen_path

    tpl = str(default_end_screen_path())
    adapter = detector_registry["pokemon_za:target_ui"](template_path=tpl)
    history = [{"score": 0.9, "sat": 200.0}] * 4
    # min_consecutive_hits defaults to 5 → 4 hits is not yet detected.
    s1 = adapter.decide(history)
    assert s1 == {"detected": False, "streak": 4}

    # Lowering min_consecutive_hits should immediately flip the state when
    # we re-decide the *same* history.
    adapter.update_params(min_consecutive_hits=3)
    s2 = adapter.decide(history)
    assert s2 == {"detected": True, "streak": 4}

    # And calling decide a third time on the same history matches the second
    # — no hidden state mutated.
    s3 = adapter.decide(history)
    assert s3 == s2
