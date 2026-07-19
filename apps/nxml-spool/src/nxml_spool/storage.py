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

import httpx

from nxml_spool.shards import Shard


@dataclass(frozen=True, slots=True)
class StorageCommit:
    commit_id: str
    shard_key: str
    checksum: str
    size_bytes: int
    receipt: dict[str, object] | None = None
    authoritative_receipt: bool = False


class StorageError(RuntimeError):
    """Base class for backend failures that must preserve local source data."""


class StorageConflictError(StorageError):
    pass


class StorageValidationError(StorageError):
    pass


class StorageUnavailableError(StorageError):
    pass


class StorageBackend(Protocol):
    def publish(self, shard: Shard) -> StorageCommit:
        """Publish an immutable shard and return only after commit verification."""

    def status(self) -> dict[str, object]: ...


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

    def status(self) -> dict[str, object]:
        return {"backend": "filesystem", "cluster_connected": None}

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

    def status(self) -> dict[str, object]:
        return {"backend": "huggingface", "cluster_connected": None}


class ControlPlaneStorageBackend:
    """REST ingest backend with an independently verified commit receipt."""

    def __init__(
        self,
        base_url: str,
        *,
        dataset_id: str,
        edge_id: str,
        token: str | None = None,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.edge_id = edge_id
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout_s,
        )

    def publish(self, shard: Shard) -> StorageCommit:
        shard_id = f"sha256:{shard.sha256}"
        object_key = f"uploads/{self.edge_id}/{shard.sha256}.tar"
        idempotency_key = f"{self.edge_id}:{self.dataset_id}:{shard.sha256}"
        upload = self._json(
            "POST",
            "/v1/uploads",
            headers={"Idempotency-Key": idempotency_key},
            json={
                "object_key": object_key,
                "size_bytes": shard.size_bytes,
                "sha256": shard.sha256,
            },
        )
        upload_id = _required_str(upload, "id")
        state = upload.get("state")
        upload_url = _required_str(upload, "upload_url")
        if state == "created":
            with shard.path.open("rb") as stream:
                self._json("PUT", upload_url, content=stream.read())

        inspected = self._json("POST", f"/v1/uploads/{upload_id}/inspect")
        if inspected.get("state") not in {"uploaded", "committed"}:
            raise StorageUnavailableError(
                f"upload {upload_id} did not become verified: {inspected.get('state')!r}"
            )
        receipt = self._json(
            "POST",
            f"/v1/uploads/{upload_id}/commit",
            json={
                "dataset_id": self.dataset_id,
                "shard_id": shard_id,
                "manifest": shard.api_manifest,
            },
        )
        commit_id = _required_str(receipt, "commit_id")
        # Commit response is not the deletion authority. Resolve the receipt
        # independently so a response-loss retry converges on catalog state.
        authoritative = self._json("GET", f"/v1/commits/{commit_id}")
        self._validate_receipt(authoritative, shard, shard_id, upload_id)
        return StorageCommit(
            commit_id,
            _required_str(authoritative, "storage_key"),
            shard.sha256,
            shard.size_bytes,
            receipt=authoritative,
            authoritative_receipt=True,
        )

    def _validate_receipt(
        self,
        receipt: dict[str, object],
        shard: Shard,
        shard_id: str,
        upload_id: str,
    ) -> None:
        expected = {
            "upload_id": upload_id,
            "checksum": shard.sha256,
            "size_bytes": shard.size_bytes,
            "state": "committed",
            "dataset_id": self.dataset_id,
            "shard_id": shard_id,
        }
        mismatches = {
            key: {"expected": value, "actual": receipt.get(key)}
            for key, value in expected.items()
            if receipt.get(key) != value
        }
        if mismatches:
            raise StorageValidationError(f"authoritative receipt mismatch: {mismatches}")

    def _json(self, method: str, path: str, **kwargs) -> dict[str, object]:
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise StorageUnavailableError(str(error)) from error
        if response.status_code == 409:
            raise StorageConflictError(_response_detail(response))
        if response.status_code == 422:
            raise StorageValidationError(_response_detail(response))
        if response.status_code >= 400:
            raise StorageUnavailableError(
                f"control plane {method} {path}: {response.status_code} {_response_detail(response)}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise StorageUnavailableError(
                f"control plane {method} {path} returned malformed JSON"
            ) from error
        if not isinstance(payload, dict):
            raise StorageUnavailableError(
                f"control plane {method} {path} returned non-object JSON"
            )
        return payload

    def status(self) -> dict[str, object]:
        try:
            health = self._json("GET", "/healthz")
            datasets = self._json("GET", "/v1/datasets").get("datasets", [])
            deployment = self._json("GET", "/v1/deployment")
        except StorageError as error:
            return {
                "backend": "control-plane",
                "cluster_connected": False,
                "cluster_error": str(error),
            }
        dataset_rows = datasets if isinstance(datasets, list) else []
        return {
            "backend": "control-plane",
            "cluster_connected": True,
            "cluster_error": None,
            "cluster_upload_counts": health,
            "dataset_count": len(dataset_rows),
            "dataset_shard_count": sum(
                int(row.get("shard_count", 0)) for row in dataset_rows if isinstance(row, dict)
            ),
            "dataset_episode_count": sum(
                int(row.get("episode_count", 0)) for row in dataset_rows if isinstance(row, dict)
            ),
            # The current API addresses snapshots by ID but does not list them.
            "snapshot_count": None,
            "active_policy_revision": deployment.get("active_revision"),
            "previous_policy_revision": deployment.get("previous_revision"),
            "deployment_generation": deployment.get("generation"),
        }


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise StorageUnavailableError(f"control-plane response missing {key!r}")
    return value


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    return str(payload.get("detail", payload)) if isinstance(payload, dict) else str(payload)
