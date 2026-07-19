#!/usr/bin/env python3
"""Read-only cradle-ns edge health probe with JSON output."""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def request_json(url: str, *, token: str = "", timeout: float = 2.0) -> dict[str, Any]:
    headers = {"X-Autopilot-Token": token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def endpoint_check(url: str, predicate, *, token: str = "") -> dict[str, Any]:
    try:
        payload = request_json(url, token=token)
        return {"ok": bool(predicate(payload)), "payload": payload}
    except (OSError, ValueError, urllib.error.URLError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def capture_check(path: Path, *, require_durable: bool) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError as error:
        return {"ok": False, "path": str(path), "error": str(error)}
    accessible = os.access(path, os.R_OK | os.W_OK)
    durable_group_member = info.st_gid in os.getgroups()
    is_device = stat.S_ISCHR(info.st_mode)
    ok = accessible and is_device and (durable_group_member or not require_durable)
    return {
        "ok": ok,
        "path": str(path),
        "character_device": is_device,
        "accessible_now": accessible,
        "durable_group_member": durable_group_member,
        "login_acl_only_risk": accessible and not durable_group_member,
    }


def spool_check(path: Path, *, max_age_s: float) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        age = max(0.0, time.time() - float(payload["updated_at"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {"ok": False, "path": str(path), "error": str(error)}
    return {"ok": age <= max_age_s, "path": str(path), "age_seconds": age, "payload": payload}


def disk_check(path: Path, *, min_free_gb: float) -> dict[str, Any]:
    try:
        stats = os.statvfs(path)
    except OSError as error:
        return {"ok": False, "path": str(path), "error": str(error)}
    free_gb = stats.f_bavail * stats.f_frsize / 1e9
    return {
        "ok": free_gb >= min_free_gb,
        "path": str(path),
        "free_gb": free_gb,
        "min_free_gb": min_free_gb,
    }


def build_report(env: dict[str, str]) -> dict[str, Any]:
    orchestrator = env.get("NXML_ORCHESTRATOR_URL", "http://127.0.0.1:7777").rstrip("/")
    autopilot = env.get("NXML_AUTOPILOT_URL", "http://127.0.0.1:8080").rstrip("/")
    token = env.get("AUTOPILOT_WEB_TOKEN", "")
    capture_dir = Path(env.get("NXML_CAPTURE_DIR", "."))
    spool_status = Path(env.get("NXML_SPOOL_STATE_DIR", ".")) / "status.json"
    checks = {
        "orchestrator": endpoint_check(
            orchestrator + "/health",
            lambda value: value.get("running") is True and value.get("connected") is True,
        ),
        "autopilot": endpoint_check(autopilot + "/health", lambda value: value.get("ok") is True),
        "runtime": endpoint_check(
            autopilot + "/runtime/status",
            lambda value: value.get("attached") is True and bool(value.get("mode")),
            token=token,
        ),
        "capture": capture_check(
            Path(env.get("NXML_CAPTURE_DEVICE", "/dev/video0")),
            require_durable=env.get("NXML_CAPTURE_REQUIRE_DURABLE_ACCESS", "1") != "0",
        ),
        "spool": spool_check(
            spool_status,
            max_age_s=float(env.get("NXML_SPOOL_STATUS_MAX_AGE_SECONDS", "60")),
        ),
        "disk": disk_check(
            capture_dir,
            min_free_gb=float(env.get("NXML_DISK_MIN_FREE_GB", "50")),
        ),
    }
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "checked_at": time.time(),
        "checks": checks,
        "missing_runtime_contracts": [
            "capture_freshness_timestamp",
            "controller_freshness_timestamp",
            "policy_endpoint_health",
            "emergency_eject_status",
            "authoritative_shard_commit_receipt",
            "active_and_previous_model_revision",
        ],
    }


def write_report(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report({**os.environ, **load_env(args.env)})
    write_report(report, args.output)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
