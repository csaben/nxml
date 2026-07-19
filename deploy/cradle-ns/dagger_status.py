"""Read-only DAgger operations status, isolated from the play data plane."""

from __future__ import annotations

import json
import os
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
        cluster_storage_path: str | None = None,
        poll_seconds: float = 5.0,
        request_timeout: float = 2.0,
    ) -> None:
        self.spool_status_path = spool_status_path
        self.cluster_url = cluster_url.rstrip("/")
        self.cluster_token_path = cluster_token_path
        self.capture_dir = capture_dir
        self.cluster_storage_path = cluster_storage_path
        self.poll_seconds = poll_seconds
        self.request_timeout = request_timeout
        self._status = OperationsStatus(observed_at=time.time())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rate_sample: tuple[float, int, int] | None = None

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
                        else next((str(item.get("error")) for item in blocked.values()), None)
                    ),
                }
            )
            value["local_storage"] = self._local_storage(value, source_bytes, staging_bytes)
            storage = value["local_storage"]
            if not storage["recording_admission_open"]:
                value["admission_open"] = False
                value["blocked_reason"] = storage["recording_blocked_reason"]
            value["rolling_segments"] = {
                "state": "disabled",
                "segment_backlog_count": None,
                "segment_backlog_bytes": None,
                "backpressure": False,
                "blocked_reason": "cluster segment routes are not active",
            }
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
            details = {}
            for job in result["jobs"].get("jobs", []):
                job_id = job.get("job_id")
                if not job_id:
                    continue
                details[job_id] = {}
                for name in ("logs", "metrics", "artifacts"):
                    request = urllib.request.Request(
                        f"{self.cluster_url}/v1/training/jobs/{job_id}/{name}", headers=headers
                    )
                    with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                        details[job_id][name] = json.load(response)
            result["job_details"] = details
            if self.cluster_storage_path is None:
                result["storage"] = {
                    "state": "unavailable",
                    "reason": "cluster storage telemetry endpoint is not published",
                }
            else:
                try:
                    request = urllib.request.Request(
                        self.cluster_url + self.cluster_storage_path, headers=headers
                    )
                    with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                        raw_storage = json.load(response)
                    filesystem = next(
                        (
                            item
                            for item in raw_storage.get("filesystems", [])
                            if item.get("role") == "authoritative_objects"
                        ),
                        None,
                    )
                    if filesystem is None or filesystem.get("status") != "available":
                        raise ValueError("cluster authoritative storage is unavailable")
                    total = int(filesystem["total_bytes"])
                    used = int(filesystem["used_bytes"])
                    result["storage"] = {
                        **raw_storage,
                        "state": "ready",
                        "total_bytes": total,
                        "used_bytes": used,
                        "free_bytes": int(filesystem["free_bytes"]),
                        "available_bytes": int(filesystem["available_bytes"]),
                        "used_fraction": used / total if total else 1.0,
                        "warning": raw_storage.get("ingest_admission", {}).get("state")
                        != "admitting",
                    }
                except (OSError, ValueError, urllib.error.URLError) as error:
                    result["storage"] = {"state": "unavailable", "reason": str(error)}
                    errors["cluster_storage"] = str(error)
            return result
        except (OSError, ValueError, urllib.error.URLError) as error:
            errors["cluster"] = str(error)
            return None

    def _local_storage(
        self, spool: dict[str, Any], source_bytes: int, staging_bytes: int
    ) -> dict[str, Any]:
        now = time.monotonic()
        try:
            capture_path = self._existing_path(self.capture_dir or self.spool_status_path.parent)
            spool_path = self._existing_path(self.spool_status_path.parent)
            capture = self._volume(capture_path)
            spool_volume = self._volume(spool_path)
            receipted_bytes = self._receipted_bytes()
            prior = self._rate_sample
            write_rate = receipt_rate = 0.0
            if prior is not None and now > prior[0]:
                elapsed = now - prior[0]
                write_rate = max(0.0, (source_bytes - prior[1]) / elapsed)
                receipt_rate = max(0.0, (receipted_bytes - prior[2]) / elapsed)
            self._rate_sample = (now, source_bytes, receipted_bytes)
            used_fraction = capture["used_fraction"]
            high = float(spool.get("disk_high_watermark", 0.85))
            low = float(spool.get("disk_low_watermark", 0.75))
            admission = used_fraction < high
            available = int(capture["available_bytes"])
            return {
                "schema_id": "nxml.edge-storage-status.v1",
                "state": "ready",
                "capture_filesystem": capture,
                "spool_filesystem": spool_volume,
                "same_filesystem": capture_path.stat().st_dev == spool_path.stat().st_dev,
                "source_buffered_bytes": source_bytes,
                "staged_bytes": staging_bytes,
                "pending_bytes": source_bytes + staging_bytes,
                "receipted_bytes": receipted_bytes,
                "capture_write_rate_bytes_per_second": write_rate,
                "upload_rate_bytes_per_second": None,
                "upload_rate_state": "unavailable_no_progress_counter",
                "receipt_rate_bytes_per_second": receipt_rate,
                "estimated_recording_seconds_remaining": (
                    available / write_rate if write_rate > 0 else None
                ),
                "warning_fraction": low,
                "fail_closed_fraction": high,
                "warning": used_fraction >= low,
                "recording_admission_open": admission,
                "recording_blocked_reason": (
                    None if admission else "capture filesystem reached fail-closed watermark"
                ),
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {
                "schema_id": "nxml.edge-storage-status.v1",
                "state": "unavailable",
                "error": str(error),
                "recording_admission_open": False,
                "recording_blocked_reason": "local storage telemetry unavailable",
            }

    @staticmethod
    def _existing_path(path: Path) -> Path:
        candidate = path.expanduser()
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        if not candidate.exists():
            raise OSError(f"no existing ancestor for {path}")
        return candidate.resolve()

    @staticmethod
    def _volume(path: Path) -> dict[str, Any]:
        stat = os.statvfs(path)
        unit = stat.f_frsize
        total = stat.f_blocks * unit
        free = stat.f_bfree * unit
        available = stat.f_bavail * unit
        used = total - free
        return {
            "path": str(path),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "available_bytes": available,
            "used_fraction": used / total if total else 1.0,
        }

    def _receipted_bytes(self) -> int:
        journal_path = self.spool_status_path.with_name("journal.json")
        if not journal_path.is_file():
            return 0
        journal = json.loads(journal_path.read_text())
        seen: set[str] = set()
        total = 0
        for item in (journal.get("shipped_episodes") or {}).values():
            receipt = item.get("receipt") or {}
            identity = str(receipt.get("shard_id") or receipt.get("commit_id") or "")
            if identity and identity not in seen:
                seen.add(identity)
                total += int(receipt.get("size_bytes") or 0)
        return total
