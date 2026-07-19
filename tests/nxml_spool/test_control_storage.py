from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from nxml_spool.shards import Shard
from nxml_spool.storage import (
    ControlPlaneStorageBackend,
    StorageConflictError,
    StorageUnavailableError,
    StorageValidationError,
)


def _shard(tmp_path: Path) -> Shard:
    path = tmp_path / "shard-000000.tar"
    path.write_bytes(b"tar-bytes")
    sidecar = tmp_path / "shard-000000.json"
    sidecar.write_text("{}")
    checksum = "a" * 64
    return Shard(
        path=path,
        sidecar_path=sidecar,
        episode_ids=["ep-1"],
        size_bytes=len(b"tar-bytes"),
        sha256=checksum,
        api_manifest={
            "schema_id": "nxml.episode.v2",
            "action_spec_id": "switch_packets.v1",
            "episodes": [{"episode_id": "ep-1"}],
            "members": [
                {"path": "ep-1.parquet", "kind": "actions", "size_bytes": 1, "sha256": "b" * 64},
                {"path": "ep-1.events.parquet", "kind": "events", "size_bytes": 1, "sha256": "c" * 64},
            ],
        },
    )


class ControlState:
    def __init__(self) -> None:
        self.state = "created"
        self.commit_id = "commit-1"
        self.commit_calls = 0
        self.upload_calls = 0
        self.lose_first_commit_response = False
        self.fail_first_upload = False
        self.commit_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/uploads" and request.method == "POST":
            return self.response(
                201,
                {
                    "id": "upload-1",
                    "upload_url": "/v1/uploads/upload-1/content",
                    "state": self.state,
                },
            )
        if path.endswith("/content"):
            self.upload_calls += 1
            if self.fail_first_upload and self.upload_calls == 1:
                raise httpx.ReadError("upload interrupted", request=request)
            self.state = "uploaded"
            return self.response(200, {"id": "upload-1", "state": self.state})
        if path.endswith("/inspect"):
            return self.response(200, {"id": "upload-1", "state": self.state})
        if path.endswith("/commit"):
            self.commit_calls += 1
            if self.commit_status != 200:
                return self.response(self.commit_status, {"detail": "rejected"})
            self.state = "committed"
            if self.lose_first_commit_response and self.commit_calls == 1:
                raise httpx.ReadError("response lost after commit", request=request)
            return self.response(200, self.receipt())
        if path == f"/v1/commits/{self.commit_id}":
            return self.response(200, self.receipt())
        if path == "/healthz":
            return self.response(200, {"status": "ok", "created": 0, "uploaded": 0, "committed": 1})
        if path == "/v1/datasets":
            return self.response(200, {"datasets": [{"id": "raw", "shard_count": 1, "episode_count": 2}]})
        if path == "/v1/deployment":
            return self.response(200, {"active_revision": "rev-2", "previous_revision": "rev-1", "generation": 3})
        return self.response(404, {"detail": "not found"})

    def receipt(self):
        return {
            "commit_id": self.commit_id,
            "upload_id": "upload-1",
            "checksum": "a" * 64,
            "size_bytes": len(b"tar-bytes"),
            "storage_key": "uploads/edge/a.tar",
            "state": "committed",
            "committed_at": "2026-07-18T00:00:00Z",
            "dataset_id": "raw",
            "shard_id": f"sha256:{'a' * 64}",
        }

    @staticmethod
    def response(status: int, payload: dict) -> httpx.Response:
        return httpx.Response(status, json=payload)


def _backend(state: ControlState) -> ControlPlaneStorageBackend:
    client = httpx.Client(base_url="http://control", transport=httpx.MockTransport(state.handler))
    return ControlPlaneStorageBackend(
        "http://control", dataset_id="raw", edge_id="edge", client=client
    )


def test_interrupted_upload_retries_same_identity(tmp_path: Path) -> None:
    state = ControlState()
    state.fail_first_upload = True
    backend = _backend(state)
    with pytest.raises(StorageUnavailableError, match="interrupted"):
        backend.publish(_shard(tmp_path))
    commit = backend.publish(_shard(tmp_path))
    assert commit.authoritative_receipt is True
    assert state.upload_calls == 2


def test_response_loss_after_commit_resolves_same_receipt(tmp_path: Path) -> None:
    state = ControlState()
    state.lose_first_commit_response = True
    backend = _backend(state)
    with pytest.raises(StorageUnavailableError, match="response lost"):
        backend.publish(_shard(tmp_path))
    commit = backend.publish(_shard(tmp_path))
    assert commit.commit_id == "commit-1"
    assert commit.receipt == state.receipt()
    assert state.commit_calls == 2
    # Further duplicate retry converges on the same immutable receipt.
    assert backend.publish(_shard(tmp_path)).receipt == commit.receipt


@pytest.mark.parametrize(
    ("status", "error"),
    [(409, StorageConflictError), (422, StorageValidationError)],
)
def test_semantic_failures_are_typed(tmp_path: Path, status: int, error: type[Exception]) -> None:
    state = ControlState()
    state.commit_status = status
    with pytest.raises(error, match="rejected"):
        _backend(state).publish(_shard(tmp_path))


def test_cluster_status_surfaces_catalog_and_deployment() -> None:
    status = _backend(ControlState()).status()
    assert status["cluster_connected"] is True
    assert status["dataset_count"] == 1
    assert status["dataset_shard_count"] == 1
    assert status["dataset_episode_count"] == 2
    assert status["snapshot_count"] is None
    assert status["active_policy_revision"] == "rev-2"
    assert status["previous_policy_revision"] == "rev-1"
