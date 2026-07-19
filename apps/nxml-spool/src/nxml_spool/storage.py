"""Durable immutable shard storage backends.

The spooler commits three objects in order: tar, metadata sidecar, commit
marker. The marker is the authoritative visibility boundary. Repeating a
publish after a crash is idempotent when all existing bytes match.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nxml_spool.shards import Shard


@dataclass(frozen=True, slots=True)
class StorageCommit:
    commit_id: str
    shard_key: str
    checksum: str
    size_bytes: int


class StorageBackend(Protocol):
    def publish(self, shard: Shard) -> StorageCommit:
        """Publish an immutable shard and return only after commit verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FilesystemStorageBackend:
    """Crash-safe backend for mounted S3/cluster object storage fixtures.

    Files are copied to a sibling temporary path, checksum-verified, then
    atomically renamed. Existing matching objects make retries no-ops;
    conflicting bytes are never overwritten.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def publish(self, shard: Shard) -> StorageCommit:
        shard_key = f"shards/{shard.path.name}"
        sidecar_key = f"meta/{shard.sidecar_path.name}"
        marker_key = f"commits/{shard.path.stem}.commit.json"
        self._put_immutable(shard.path, shard_key, shard.sha256)
        self._put_immutable(
            shard.sidecar_path,
            sidecar_key,
            sha256_file(shard.sidecar_path),
        )
        marker = {
            "commit_id": shard.sha256,
            "shard_key": shard_key,
            "metadata_key": sidecar_key,
            "sha256": shard.sha256,
            "bytes": shard.size_bytes,
        }
        marker_bytes = json.dumps(marker, sort_keys=True).encode()
        self._put_bytes_immutable(marker_bytes, marker_key)
        return StorageCommit(shard.sha256, shard_key, shard.sha256, shard.size_bytes)

    def _put_immutable(self, source: Path, key: str, checksum: str) -> None:
        destination = self.root / key
        if destination.exists():
            if sha256_file(destination) != checksum:
                raise RuntimeError(f"immutable object conflict: {key}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != checksum:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"staged checksum mismatch: {key}")
        temporary.replace(destination)

    def _put_bytes_immutable(self, payload: bytes, key: str) -> None:
        destination = self.root / key
        if destination.exists():
            if destination.read_bytes() != payload:
                raise RuntimeError(f"immutable commit conflict: {key}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)


class HFDatasetStorageBackend:
    """Compatibility backend for existing Hugging Face dataset repos."""

    def __init__(self, repo_id: str, *, private: bool = True) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id
        self.api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    def _upload_and_size_verify(self, local_path: Path, key: str) -> None:
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=key,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        info = self.api.get_paths_info(self.repo_id, [key], repo_type="dataset")
        remote_size = info[0].size if info else None
        if remote_size != local_path.stat().st_size:
            raise RuntimeError(
                f"upload verification failed for {key}: "
                f"remote={remote_size} local={local_path.stat().st_size}"
            )

    def publish(self, shard: Shard) -> StorageCommit:
        self._upload_and_size_verify(shard.path, f"shards/{shard.path.name}")
        self._upload_and_size_verify(shard.sidecar_path, f"meta/{shard.sidecar_path.name}")
        marker_path = shard.sidecar_path.with_suffix(".commit.tmp")
        marker_path.write_text(
            json.dumps(
                {
                    "commit_id": shard.sha256,
                    "shard_key": f"shards/{shard.path.name}",
                    "sha256": shard.sha256,
                    "bytes": shard.size_bytes,
                },
                sort_keys=True,
            )
        )
        try:
            self._upload_and_size_verify(
                marker_path, f"commits/{shard.path.stem}.commit.json"
            )
        finally:
            marker_path.unlink(missing_ok=True)
        return StorageCommit(
            shard.sha256,
            f"shards/{shard.path.name}",
            shard.sha256,
            shard.size_bytes,
        )
