from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path


def _load_health_module():
    path = Path(__file__).parents[2] / "deploy/cradle-ns/nxml_edge_health.py"
    spec = importlib.util.spec_from_file_location("nxml_edge_health_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_spool_status_is_degraded(tmp_path: Path) -> None:
    health = _load_health_module()
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"updated_at": time.time() - 120}))
    result = health.spool_check(status, max_age_s=60)
    assert result["ok"] is False
    assert result["age_seconds"] >= 120


def test_login_acl_without_device_group_is_not_durable(monkeypatch) -> None:
    health = _load_health_module()
    monkeypatch.setattr(os, "access", lambda *_args: True)
    monkeypatch.setattr(os, "getgroups", lambda: [])
    result = health.capture_check(Path("/dev/null"), require_durable=True)
    assert result["accessible_now"] is True
    assert result["durable_group_member"] is False
    assert result["login_acl_only_risk"] is True
    assert result["ok"] is False


def test_endpoint_predicate_rejects_disconnected_orchestrator(monkeypatch) -> None:
    health = _load_health_module()
    monkeypatch.setattr(
        health,
        "request_json",
        lambda *_args, **_kwargs: {"running": True, "connected": False},
    )
    result = health.endpoint_check(
        "http://fixture/health",
        lambda payload: payload["running"] and payload["connected"],
    )
    assert result["ok"] is False
