import hashlib
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.catalog import Catalog
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage

from tests.nxml_control.fixture_shard import fixture_shard


def shard_manifest(episode_id="e1"):
    return {
        "schema_id": "nxml.episode.v2",
        "action_spec_id": "switch_packets.v1",
        "episodes": [{"episode_id": episode_id}],
        "members": [
            {
                "path": f"{episode_id}.parquet",
                "kind": "actions",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
            {
                "path": f"{episode_id}.events.parquet",
                "kind": "events",
                "size_bytes": 1,
                "sha256": "b" * 64,
            },
        ],
    }


def components(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    return catalog, IngestService(catalog, LocalObjectStorage(tmp_path / "objects"))


def test_checksum_gated_exactly_once_commit(tmp_path):
    catalog, service = components(tmp_path)
    content = b"immutable webdataset"
    digest = hashlib.sha256(content).hexdigest()
    request = dict(
        idempotency_key="capture/shard",
        object_key="uploads/capture/shard.tar",
        size_bytes=len(content),
        sha256=digest,
    )
    created = catalog.create_upload(**request)
    assert catalog.create_upload(**request).id == created.id
    with pytest.raises(ValueError, match="verified"):
        catalog.commit(created.id, dataset_id="raw", shard_id="s1", manifest=shard_manifest())
    assert service.upload(created.id, BytesIO(content)).state == "uploaded"
    committed = catalog.commit(
        created.id, dataset_id="raw", shard_id="s1", manifest=shard_manifest()
    )
    assert committed.state == "committed"
    assert (
        catalog.commit(created.id, dataset_id="raw", shard_id="s1", manifest=shard_manifest()).id
        == committed.id
    )


def test_changed_idempotent_request_and_bad_checksum_fail(tmp_path):
    catalog, service = components(tmp_path)
    upload = catalog.create_upload(
        idempotency_key="same", object_key="uploads/a.tar", size_bytes=3, sha256="0" * 64
    )
    with pytest.raises(ValueError, match="different request"):
        catalog.create_upload(
            idempotency_key="same", object_key="uploads/a.tar", size_bytes=4, sha256="1" * 64
        )
    with pytest.raises(ValueError, match="checksum"):
        service.upload(upload.id, BytesIO(b"bad"))
    assert catalog.get(upload.id).state == "created"


def test_dagger_episode_requires_explicit_positive_quality_before_snapshot(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, manifest = fixture_shard("dagger-episode")
    manifest["episodes"][0]["action_schema_id"] = "nxml.dagger-actions.v2"
    digest = hashlib.sha256(content).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "dagger-explicit-quality"},
        json={
            "object_key": "uploads/dagger-explicit-quality.tar",
            "size_bytes": len(content),
            "sha256": digest,
        },
    ).json()
    assert client.put(upload["upload_url"], content=content).status_code == 200
    assert client.post(
        f"/v1/uploads/{upload['id']}/commit",
        json={"dataset_id": "dagger", "shard_id": "dagger-1", "manifest": manifest},
    ).status_code == 200

    missing = client.post(
        "/v1/datasets/dagger/snapshots", json={"control_source": "human"}
    )
    assert missing.status_code == 422
    quality = client.post(
        "/v1/datasets/dagger/episodes/dagger-episode/quality-dispositions",
        headers={"Idempotency-Key": "dagger-episode-validator-v2"},
        json={
            "schema_id": "nxml.episode-quality.v1",
            "training_eligible": True,
            "reason": "dagger_actions_v2_validated",
            "validator": "dagger-actions-validator",
            "validator_version": "2",
        },
    )
    assert quality.status_code == 201
    snapshot = client.post(
        "/v1/datasets/dagger/snapshots", json={"control_source": "human"}
    )
    assert snapshot.status_code == 201
    assert snapshot.json()["manifest"]["shards"][0]["episode_ids"] == ["dagger-episode"]


def test_http_contract_and_openapi(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, api_manifest = fixture_shard()
    digest = hashlib.sha256(content).hexdigest()
    response = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "edge/shard"},
        json={"object_key": "uploads/edge/shard.tar", "size_bytes": len(content), "sha256": digest},
    )
    assert response.status_code == 201
    upload = response.json()
    assert client.put(upload["upload_url"], content=content).json()["state"] == "uploaded"
    assert (
        client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "raw", "shard_id": "edge-1", "manifest": api_manifest},
        ).json()["state"]
        == "committed"
    )
    assert client.get("/healthz").json() == {"status": "ready"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/uploads/{upload_id}/inspect" in paths


def test_existing_object_key_is_immutable(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    first = storage.put_if_absent("uploads/a.tar", BytesIO(b"first"))
    second = storage.put_if_absent("uploads/a.tar", BytesIO(b"second"))
    assert second == first
    with storage.open("uploads/a.tar") as source:
        assert source.read() == b"first"


def test_normalized_catalog_and_query_endpoints(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, api_manifest = fixture_shard("ep-b")
    digest = hashlib.sha256(content).hexdigest()
    created = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "catalog/shard"},
        json={
            "object_key": "uploads/catalog/shard.tar",
            "size_bytes": len(content),
            "sha256": digest,
        },
    ).json()
    client.put(created["upload_url"], content=content)
    manifest = api_manifest
    assert (
        client.post(
            f"/v1/uploads/{created['id']}/commit",
            json={"dataset_id": "raw-v2", "shard_id": "shard-1", "manifest": manifest},
        ).status_code
        == 200
    )
    assert client.get("/v1/datasets").json() == {
        "datasets": [{"id": "raw-v2", "shard_count": 1, "episode_count": 1}]
    }
    shards = client.get("/v1/datasets/raw-v2/shards").json()["shards"]
    assert shards[0]["sha256"] == digest
    episodes = client.get("/v1/datasets/raw-v2/episodes").json()["episodes"]
    assert [item["id"] for item in episodes] == ["ep-b"]


def test_committed_manifest_is_immutable(tmp_path):
    catalog, service = components(tmp_path)
    content = b"x"
    digest = hashlib.sha256(content).hexdigest()
    upload = catalog.create_upload(
        idempotency_key="x", object_key="uploads/x.tar", size_bytes=1, sha256=digest
    )
    service.upload(upload.id, BytesIO(content))
    catalog.commit(upload.id, dataset_id="raw", shard_id="s", manifest=shard_manifest("e"))
    with pytest.raises(ValueError, match="manifest cannot be changed"):
        catalog.commit(upload.id, dataset_id="raw", shard_id="s", manifest=shard_manifest("other"))


def test_receipt_is_immutable_queryable_and_snapshot_is_content_addressed(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, api_manifest = fixture_shard()
    digest = hashlib.sha256(content).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "receipt"},
        json={"object_key": "uploads/receipt.tar", "size_bytes": len(content), "sha256": digest},
    ).json()
    client.put(upload["upload_url"], content=content)
    request = {"dataset_id": "raw", "shard_id": "receipt-1", "manifest": api_manifest}
    receipt = client.post(f"/v1/uploads/{upload['id']}/commit", json=request).json()
    assert set(receipt) == {
        "commit_id",
        "upload_id",
        "checksum",
        "size_bytes",
        "storage_key",
        "state",
        "committed_at",
        "dataset_id",
        "shard_id",
    }
    assert receipt["checksum"] == digest and receipt["state"] == "committed"
    assert client.post(f"/v1/uploads/{upload['id']}/commit", json=request).json() == receipt
    assert client.get(f"/v1/commits/{receipt['commit_id']}").json() == receipt
    first = client.post("/v1/datasets/raw/snapshots", json={"control_source": "human"}).json()
    second = client.post("/v1/datasets/raw/snapshots", json={"control_source": "human"}).json()
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["manifest"]["shards"][0]["sha256"] == digest
    assert (
        client.get(f"/v1/snapshots/{first['snapshot_id']}").json()["manifest"] == first["manifest"]
    )


def test_manifest_validation_is_422_and_identity_conflict_is_409(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, api_manifest = fixture_shard()
    digest = hashlib.sha256(content).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "status"},
        json={"object_key": "uploads/status.tar", "size_bytes": len(content), "sha256": digest},
    ).json()
    client.put(upload["upload_url"], content=content)
    assert (
        client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "raw", "shard_id": "s", "manifest": {}},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "raw", "shard_id": "s", "manifest": api_manifest},
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "other", "shard_id": "s", "manifest": api_manifest},
        ).status_code
        == 409
    )
