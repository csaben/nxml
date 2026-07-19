import hashlib

from fastapi.testclient import TestClient
from nxml_control.api import create_app

from tests.nxml_control.fixture_shard import fixture_shard


def committed_episode(client):
    shard, manifest = fixture_shard("demo")
    digest = hashlib.sha256(shard).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "search-shard"},
        json={"object_key": "uploads/search.tar", "size_bytes": len(shard), "sha256": digest},
    ).json()
    # Search is strictly downstream: episode is absent before commit.
    premature = {
        "clip": {"dataset_id": "raw", "episode_id": "demo"},
        "artifact_type": "thumbnail",
        "source_snapshot_id": "missing",
        "index_version": "missing",
    }
    assert (
        client.post(
            "/v1/search/artifacts", headers={"Idempotency-Key": "premature"}, json=premature
        ).status_code
        == 404
    )
    client.put(upload["upload_url"], content=shard)
    receipt_response = client.post(
        f"/v1/uploads/{upload['id']}/commit",
        json={"dataset_id": "raw", "shard_id": "s", "manifest": manifest},
    )
    assert receipt_response.status_code == 200, receipt_response.text
    receipt = receipt_response.json()
    snapshot_response = client.post("/v1/datasets/raw/snapshots", json={"control_source": "human"})
    assert snapshot_response.status_code == 201, snapshot_response.text
    snapshot = snapshot_response.json()
    return receipt, snapshot


def test_index_artifact_annotation_contract_and_queries(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    receipt, snapshot = committed_episode(client)
    index_body = {
        "source_snapshot_id": snapshot["snapshot_id"],
        "embedding_model": None,
        "embedding_version": None,
    }
    response = client.post(
        "/v1/search/indexes", headers={"Idempotency-Key": "index-v1"}, json=index_body
    )
    assert response.status_code == 201, response.text
    index = response.json()
    assert index["state"] == "pending"
    assert (
        client.post(
            "/v1/search/indexes", headers={"Idempotency-Key": "index-v1"}, json=index_body
        ).json()
        == index
    )
    running = client.post(
        f"/v1/search/indexes/{index['index_version']}/state", json={"state": "running"}
    ).json()
    assert running["state"] == "running"
    complete = client.post(
        f"/v1/search/indexes/{index['index_version']}/state", json={"state": "complete"}
    ).json()
    assert complete["state"] == "complete"
    assert (
        client.post(
            f"/v1/search/indexes/{index['index_version']}/state", json={"state": "running"}
        ).status_code
        == 409
    )

    artifact_body = {
        "clip": {
            "dataset_id": "raw",
            "episode_id": "demo",
            "window_start_ns": 10,
            "window_end_ns": 20,
        },
        "artifact_type": "embedding",
        "embedding_model": "future/model",
        "embedding_version": "v1",
        "labels": ["battle"],
        "derived_features": {"motion": 0.4},
        "source_snapshot_id": snapshot["snapshot_id"],
        "index_version": index["index_version"],
        "status": "pending",
    }
    artifact = client.post(
        "/v1/search/artifacts", headers={"Idempotency-Key": "artifact-1"}, json=artifact_body
    ).json()
    assert artifact["window_start_ns"] == 10 and artifact["derived_features"] == {"motion": 0.4}
    assert (
        client.post(
            "/v1/search/artifacts", headers={"Idempotency-Key": "artifact-1"}, json=artifact_body
        ).json()["artifact_id"]
        == artifact["artifact_id"]
    )
    running_artifact = client.post(
        f"/v1/search/artifacts/{artifact['artifact_id']}/state", json={"status": "running"}
    ).json()
    assert running_artifact["status"] == "running"
    complete_artifact = client.post(
        f"/v1/search/artifacts/{artifact['artifact_id']}/state", json={"status": "complete"}
    ).json()
    assert complete_artifact["status"] == "complete"
    assert (
        client.get("/v1/search/episodes/raw/demo/artifacts").json()["artifacts"][0]["artifact_id"]
        == artifact["artifact_id"]
    )

    annotation_body = {
        "clip": {
            "dataset_id": "raw",
            "episode_id": "demo",
            "window_start_ns": 10,
            "window_end_ns": 20,
        },
        "labels": ["boss", "human-demo"],
        "source_snapshot_id": snapshot["snapshot_id"],
        "index_version": index["index_version"],
        "status": "complete",
    }
    annotation = client.post(
        "/v1/search/annotations", headers={"Idempotency-Key": "annotation-1"}, json=annotation_body
    ).json()
    assert annotation["labels"] == ["boss", "human-demo"]
    assert (
        client.get("/v1/search/episodes/raw/demo/annotations").json()["annotations"][0][
            "annotation_id"
        ]
        == annotation["annotation_id"]
    )
    bad = {
        **artifact_body,
        "clip": {
            "dataset_id": "raw",
            "episode_id": "demo",
            "window_start_ns": 20,
            "window_end_ns": 10,
        },
    }
    assert (
        client.post(
            "/v1/search/artifacts", headers={"Idempotency-Key": "bad-clip"}, json=bad
        ).status_code
        == 409
    )
    # Downstream metadata never changes the authoritative receipt.
    assert client.get(f"/v1/commits/{receipt['commit_id']}").json() == receipt


def test_failed_index_state_and_idempotency_conflicts(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    _, snapshot = committed_episode(client)
    body = {"source_snapshot_id": snapshot["snapshot_id"]}
    response = client.post(
        "/v1/search/indexes", headers={"Idempotency-Key": "failed-index"}, json=body
    )
    assert response.status_code == 201, response.text
    index = response.json()
    failed = client.post(
        f"/v1/search/indexes/{index['index_version']}/state",
        json={"state": "failed", "error": "worker unavailable"},
    ).json()
    assert failed["state"] == "failed" and failed["error"] == "worker unavailable"
    conflict = client.post(
        "/v1/search/indexes",
        headers={"Idempotency-Key": "failed-index"},
        json={**body, "embedding_model": "changed"},
    )
    assert conflict.status_code == 409
