"""Read-only DAgger operations status, isolated from the play data plane."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionStatus:
    schema_version: str = "nxml.dagger-session-status.v1"
    mode: str = "human"
    recording_state: str = "idle"
    recording_available: bool = False
    recording_blocked_reason: str = (
        "Live preview and recording currently require exclusive access to the capture device"
    )


@dataclass(frozen=True)
class OperationsStatus:
    schema_version: str = "nxml.dagger-operations-status.v1"
    observed_at: float = 0.0
    session: SessionStatus = field(default_factory=SessionStatus)
    spool: dict[str, Any] | None = None
    cluster: dict[str, Any] | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def wire(self) -> dict[str, Any]:
        return asdict(self)


class OperationsReader:
    """Poll slow local/cluster state on its own thread and publish cached snapshots."""

    def __init__(
        self,
        *,
        spool_status_path: Path,
        cluster_url: str,
        cluster_token_path: Path | None,
        capture_dir: Path | None = None,
        poll_seconds: float = 5.0,
        request_timeout: float = 2.0,
    ) -> None:
        self.spool_status_path = spool_status_path
        self.cluster_url = cluster_url.rstrip("/")
        self.cluster_token_path = cluster_token_path
        self.capture_dir = capture_dir
        self.poll_seconds = poll_seconds
        self.request_timeout = request_timeout
        self._status = OperationsStatus(observed_at=time.time())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="dagger-ops-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.request_timeout * 2, 1.0))

    def snapshot(self) -> OperationsStatus:
        with self._lock:
            return self._status

    def refresh(self) -> OperationsStatus:
        errors: dict[str, str] = {}
        spool = self._read_spool(errors)
        cluster = self._read_cluster(errors)
        status = OperationsStatus(
            observed_at=time.time(), spool=spool, cluster=cluster, errors=errors
        )
        with self._lock:
            self._status = status
        return status

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.poll_seconds)

    def _read_spool(self, errors: dict[str, str]) -> dict[str, Any] | None:
        try:
            value = json.loads(self.spool_status_path.read_text())
            if not isinstance(value, dict):
                raise ValueError("status root is not an object")
            staging_bytes = int(value.get("staging_bytes") or 0)
            source_files = (
                [path for path in self.capture_dir.glob("*") if path.is_file()]
                if self.capture_dir is not None and self.capture_dir.is_dir()
                else []
            )
            source_bytes = sum(path.stat().st_size for path in source_files)
            manifests = [path for path in source_files if path.name.endswith(".manifest.json")]
            oldest_mtime = min((path.stat().st_mtime for path in manifests), default=None)
            blocked = value.get("blocked_episodes") or {}
            uploading = value.get("uploading")
            pending = int(value.get("pending_episodes") or 0)
            shipped = int(value.get("episodes_shipped") or 0)
            value.update(
                {
                    "source_buffered_bytes": source_bytes,
                    "local_buffered_bytes": source_bytes + staging_bytes,
                    "oldest_pending_age_seconds": (
                        max(0.0, time.time() - oldest_mtime) if oldest_mtime is not None else None
                    ),
                    "receipt_state": (
                        "uploading"
                        if uploading
                        else "awaiting_receipt"
                        if pending
                        else "committed_and_deleted"
                        if shipped
                        else "none"
                    ),
                    "deletion_eligible": False,
                    "blocked_reason": (
                        "disk pressure"
                        if value.get("disk_pressure")
                        else next(
                            (str(item.get("error")) for item in blocked.values()), None
                        )
                    ),
                }
            )
            return value
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors["spool"] = str(error)
            return None

    def _read_cluster(self, errors: dict[str, str]) -> dict[str, Any] | None:
        try:
            headers: dict[str, str] = {}
            if self.cluster_token_path is not None:
                token = self.cluster_token_path.read_text().strip()
                headers["Authorization"] = f"Bearer {token}"
            result: dict[str, Any] = {}
            for name, path in (
                ("datasets", "/v1/datasets"),
                ("snapshots", "/v1/snapshots?limit=20"),
                ("jobs", "/v1/training/jobs?limit=20"),
                ("revisions", "/v1/models/revisions?limit=20"),
                ("deployment", "/v1/deployment"),
            ):
                request = urllib.request.Request(self.cluster_url + path, headers=headers)
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    result[name] = json.load(response)
            return result
        except (OSError, ValueError, urllib.error.URLError) as error:
            errors["cluster"] = str(error)
            return None
