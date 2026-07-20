from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "deploy" / "cradle-ns" / "latency_probe.py"
SPEC = importlib.util.spec_from_file_location("latency_probe_safety", PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_probe_refuses_to_overwrite_held_human_action_when_minui_is_live(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    response = Response()
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *_a, **_k: response)
    monkeypatch.setattr(probe.json, "load", lambda _response: {
        "action_plane": {"mode": "human", "armed": False}
    })
    args = Namespace(allow_live_output=False, minui_status_url="http://minui/status")

    with pytest.raises(SystemExit, match="refusing competing controller writer"):
        probe.refuse_competing_writer(args)


def test_explicit_physical_override_is_required_for_competing_writer():
    args = Namespace(allow_live_output=True, minui_status_url="http://minui/status")
    probe.refuse_competing_writer(args)
