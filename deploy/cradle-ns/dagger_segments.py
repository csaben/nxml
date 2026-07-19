"""Bounded rolling-segment packaging and receipt-gated cluster delivery."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class SegmentSource:
    episode_id: str
    sequence_index: int
    timeline_start_ns: int
    timeline_end_ns: int
    video: Path
    actions: Path
    events: Path
    manifest: Path | None = None


@dataclass(frozen=True)
class PreparedSegment:
    source: SegmentSource
    tar_path: Path
    bundle: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_segment(source: SegmentSource, staging_dir: Path, dataset_id: str) -> PreparedSegment:
    if source.timeline_end_ns <= source.timeline_start_ns:
        raise ValueError("segment timeline must be nonempty and half-open")
    expected = {
        "video": f"{source.episode_id}.{source.sequence_index:06d}.mkv",
        "actions": f"{source.episode_id}.{source.sequence_index:06d}.parquet",
        "events": f"{source.episode_id}.{source.sequence_index:06d}.events.parquet",
    }
    files = {"video": source.video, "actions": source.actions, "events": source.events}
    for role, path in files.items():
        if not path.is_file() or path.name != expected[role]:
            raise ValueError(f"invalid {role} segment member")
    members = [
        {
            "role": role,
            "path": expected[role],
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in files.items()
    ]
    staging_dir.mkdir(parents=True, exist_ok=True)
    temporary = staging_dir / f".{source.episode_id}.{source.sequence_index:06d}.tar.tmp"
    with tarfile.open(temporary, "w") as archive:
        for member in members:
            path = files[member["role"]]
            info = archive.gettarinfo(str(path), arcname=member["path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    object_sha256 = _sha256(temporary)
    tar_path = staging_dir / f"{source.sequence_index:06d}-{object_sha256}.tar"
    temporary.replace(tar_path)
    bundle = {
        "schema_id": "nxml.segment-bundle.v1",
        "dataset_id": dataset_id,
        "episode_id": source.episode_id,
        "segment_id": f"sha256:{object_sha256}",
        "sequence_index": source.sequence_index,
        "clock_id": "linux-monotonic",
        "timeline_start_ns": source.timeline_start_ns,
        "timeline_end_ns": source.timeline_end_ns,
        "object_size_bytes": tar_path.stat().st_size,
        "object_sha256": object_sha256,
        "members": members,
    }
    return PreparedSegment(source, tar_path, bundle)


class SegmentClient:
    def __init__(self, base_url: str, token: str, dataset_id: str, *, timeout: float = 60):
        self.dataset_id = dataset_id
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def _json(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("cluster response is not an object")
        return value

    def publish(self, prepared: PreparedSegment) -> dict[str, Any]:
        bundle, path = prepared.bundle, prepared.tar_path
        digest = bundle["object_sha256"]
        episode_id = bundle["episode_id"]
        index = bundle["sequence_index"]
        key = f"segments:{episode_id}:{index}:{digest}"
        upload = self._json(
            "POST",
            "/v1/uploads",
            headers={"Idempotency-Key": key},
            json={
                "object_key": f"uploads/segments/{episode_id}/{index:06d}-{digest}.tar",
                "size_bytes": bundle["object_size_bytes"],
                "sha256": digest,
            },
        )
        upload_id = upload["id"]
        if upload["state"] == "created":
            with path.open("rb") as stream:
                self._json("PUT", upload["upload_url"], content=stream)
        inspected = self._json("POST", f"/v1/uploads/{upload_id}/inspect")
        if inspected["state"] not in {"uploaded", "committed"}:
            raise RuntimeError("segment upload did not verify")
        receipt = self._json("POST", f"/v1/segment-bundles/{upload_id}/commit", json=bundle)
        authoritative = self._json("GET", f"/v1/segment-receipts/{receipt['receipt_id']}")
        expected = {
            "segment_id": bundle["segment_id"],
            "upload_id": upload_id,
            "dataset_id": bundle["dataset_id"],
            "episode_id": episode_id,
            "sequence_index": index,
            "size_bytes": bundle["object_size_bytes"],
            "sha256": digest,
            "timeline_start_ns": bundle["timeline_start_ns"],
            "timeline_end_ns": bundle["timeline_end_ns"],
            "state": "committed",
        }
        mismatch = {
            k: (v, authoritative.get(k)) for k, v in expected.items() if authoritative.get(k) != v
        }
        if mismatch:
            raise RuntimeError(f"authoritative segment receipt mismatch: {mismatch}")
        return authoritative

    def close_episode(self, episode_id: str, receipts: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = sorted(receipts, key=lambda item: item["sequence_index"])
        if not ordered:
            raise ValueError("cannot close an episode without receipted segments")
        for expected_index, item in enumerate(ordered):
            if item["sequence_index"] != expected_index:
                raise ValueError("episode segments must be ordered and gap-free")
            if (
                expected_index
                and ordered[expected_index - 1]["timeline_end_ns"] != item["timeline_start_ns"]
            ):
                raise ValueError("episode segment timelines must be contiguous and half-open")
        manifest = {
            "schema_id": "nxml.episode-close.v1",
            "dataset_id": self.dataset_id,
            "episode_id": episode_id,
            "clock_id": "linux-monotonic",
            "timeline_start_ns": ordered[0]["timeline_start_ns"],
            "timeline_end_ns": ordered[-1]["timeline_end_ns"],
            "segments": [
                {
                    "segment_id": item["segment_id"],
                    "sequence_index": item["sequence_index"],
                    "timeline_start_ns": item["timeline_start_ns"],
                    "timeline_end_ns": item["timeline_end_ns"],
                    "object_sha256": item["sha256"],
                }
                for item in ordered
            ],
        }
        return self._json(
            "POST",
            f"/v1/datasets/{self.dataset_id}/episodes/{episode_id}/close",
            headers={"Idempotency-Key": f"episode-close:{episode_id}"},
            json=manifest,
        )


class SegmentJournal:
    def __init__(self, path: Path):
        self.path = path
        self.value = (
            json.loads(path.read_text())
            if path.is_file()
            else {"pending": {}, "cleanup": {}, "receipts": {}, "close": None}
        )
        self.value.setdefault("pending", {})
        self.value.setdefault("cleanup", {})

    def store_pending(self, prepared: PreparedSegment) -> None:
        source = prepared.source
        self.value["pending"][str(source.sequence_index)] = {
            "source": self._source_wire(source),
            "tar_path": str(prepared.tar_path),
            "bundle": prepared.bundle,
        }
        self._sync()

    def store_source(self, source: SegmentSource) -> None:
        self.value["pending"][str(source.sequence_index)] = {
            "source": self._source_wire(source),
            "tar_path": None,
            "bundle": None,
        }
        self._sync()

    @staticmethod
    def _source_wire(source: SegmentSource) -> dict[str, Any]:
        return {
            "episode_id": source.episode_id,
            "sequence_index": source.sequence_index,
            "timeline_start_ns": source.timeline_start_ns,
            "timeline_end_ns": source.timeline_end_ns,
            "video": str(source.video),
            "actions": str(source.actions),
            "events": str(source.events),
            "manifest": str(source.manifest) if source.manifest else None,
        }

    def pending(self) -> list[PreparedSegment]:
        result = []
        for value in self.value["pending"].values():
            if value.get("bundle") is None:
                continue
            raw = value["source"]
            source = SegmentSource(
                episode_id=raw["episode_id"],
                sequence_index=raw["sequence_index"],
                timeline_start_ns=raw["timeline_start_ns"],
                timeline_end_ns=raw["timeline_end_ns"],
                video=Path(raw["video"]),
                actions=Path(raw["actions"]),
                events=Path(raw["events"]),
                manifest=Path(raw["manifest"]) if raw.get("manifest") else None,
            )
            result.append(PreparedSegment(source, Path(value["tar_path"]), value["bundle"]))
        return sorted(result, key=lambda item: item.source.sequence_index)

    def pending_sources(self) -> list[SegmentSource]:
        result = []
        for value in self.value["pending"].values():
            if value.get("bundle") is not None:
                continue
            raw = value["source"]
            result.append(
                SegmentSource(
                    episode_id=raw["episode_id"],
                    sequence_index=raw["sequence_index"],
                    timeline_start_ns=raw["timeline_start_ns"],
                    timeline_end_ns=raw["timeline_end_ns"],
                    video=Path(raw["video"]),
                    actions=Path(raw["actions"]),
                    events=Path(raw["events"]),
                    manifest=Path(raw["manifest"]) if raw.get("manifest") else None,
                )
            )
        return sorted(result, key=lambda item: item.sequence_index)

    def store_receipt(self, receipt: dict[str, Any]) -> None:
        key = str(receipt["sequence_index"])
        pending = self.value["pending"].get(key)
        self.value["receipts"][str(receipt["sequence_index"])] = receipt
        if pending is not None:
            self.value["cleanup"][key] = pending
        self.value["pending"].pop(key, None)
        self._sync()

    def cleanup_paths(self) -> list[tuple[str, list[Path]]]:
        result = []
        for key, item in self.value["cleanup"].items():
            raw = item["source"]
            paths = [Path(raw[name]) for name in ("video", "actions", "events")]
            if raw.get("manifest"):
                paths.append(Path(raw["manifest"]))
            paths.append(Path(item["tar_path"]))
            result.append((key, paths))
        return result

    def mark_deleted(self, sequence_index: str | int) -> None:
        self.value["cleanup"].pop(str(sequence_index), None)
        self._sync()

    def store_close(self, close: dict[str, Any]) -> None:
        self.value["close"] = close
        self._sync()

    def _sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(self.value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


class SegmentDeliveryWorker:
    """Bounded async upload owner; capture only performs nonblocking submit."""

    def __init__(
        self,
        client: SegmentClient,
        journal: SegmentJournal,
        *,
        staging_dir: Path | None = None,
        max_pending: int = 3,
    ):
        self.client, self.journal = client, journal
        self.staging_dir = staging_dir
        self.queue: queue.Queue[PreparedSegment | SegmentSource | None] = queue.Queue(max_pending)
        self.receipts: list[dict[str, Any]] = list(journal.value["receipts"].values())
        self.error: str | None = None
        self.uploaded_bytes = 0
        self.started_at = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active: PreparedSegment | SegmentSource | None = None

    def start(self) -> None:
        if self._thread is None:
            for key, paths in self.journal.cleanup_paths():
                for path in paths:
                    path.unlink(missing_ok=True)
                self.journal.mark_deleted(key)
            for item in self.journal.pending_sources():
                self.queue.put_nowait(item)
            for item in self.journal.pending():
                self.queue.put_nowait(item)
            self._thread = threading.Thread(target=self._run, daemon=True, name="segment-delivery")
            self._thread.start()

    def submit(self, item: PreparedSegment) -> bool:
        if self.queue.full():
            return False
        try:
            self.journal.store_pending(item)
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def submit_source(self, source: SegmentSource) -> bool:
        if self.queue.full():
            return False
        try:
            self.journal.store_source(source)
            self.queue.put_nowait(source)
            return True
        except queue.Full:
            return False

    def stop(self, *, timeout: float = 30.0) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                return
            if isinstance(item, SegmentSource):
                if self.staging_dir is None:
                    self.error = "rolling segment staging directory is not configured"
                    return
                try:
                    item = prepare_segment(item, self.staging_dir, self.client.dataset_id)
                    self.journal.store_pending(item)
                except Exception as error:
                    self.error = str(error)
                    return
            with self._lock:
                self._active = item
            while not self._stop.is_set():
                try:
                    receipt = self.client.publish(item)
                    size = item.tar_path.stat().st_size
                    self.journal.store_receipt(receipt)
                    with self._lock:
                        self.receipts.append(receipt)
                    self.uploaded_bytes += size
                    for path in (
                        item.source.video,
                        item.source.actions,
                        item.source.events,
                        item.source.manifest,
                        item.tar_path,
                    ):
                        if path is not None:
                            path.unlink(missing_ok=True)
                    self.journal.mark_deleted(item.source.sequence_index)
                    self.error = None
                    break
                except Exception as error:
                    self.error = str(error)
                    # The item remains durable and retries with the same keys.
                    if self._stop.wait(1.0):
                        return
            with self._lock:
                self._active = None

    def wait_receipts(self, count: int, *, timeout: float = 120.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                receipts = list(self.receipts)
            if len(receipts) >= count:
                return sorted(receipts, key=lambda item: item["sequence_index"])
            time.sleep(0.05)
        raise TimeoutError("segment receipts did not become durable before close timeout")

    def close_episode(self, episode_id: str, count: int) -> dict[str, Any]:
        receipts = self.wait_receipts(count)
        close = self.client.close_episode(episode_id, receipts)
        self.journal.store_close(close)
        return close

    def status(self) -> dict[str, Any]:
        with self.queue.mutex:
            queued = list(self.queue.queue)
        with self._lock:
            active = self._active
            receipt_count = len(self.receipts)
        return {
            "state": "error" if self.error else "running",
            "segment_backlog_count": sum(item is not None for item in queued),
            "segment_backlog_bytes": sum(
                (
                    item.tar_path.stat().st_size
                    if isinstance(item, PreparedSegment) and item.tar_path.exists()
                    else sum(
                        path.stat().st_size
                        for path in (item.video, item.actions, item.events)
                        if path.exists()
                    )
                    if isinstance(item, SegmentSource)
                    else 0
                )
                for item in queued
                if item is not None
            ),
            "backpressure": self.queue.full(),
            "blocked_reason": self.error,
            "active_sequence_index": (
                active.source.sequence_index
                if isinstance(active, PreparedSegment)
                else active.sequence_index
                if isinstance(active, SegmentSource)
                else None
            ),
            "receipted_segments": receipt_count,
            "receipted_bytes": self.uploaded_bytes,
        }
