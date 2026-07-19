import hashlib

from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.materialize import iter_webdataset_rows
from nxml_control.models import FakePolicyRuntime

from tests.nxml_control.fixture_shard import fixture_shard


def test_milestones_1_to_3_completion_chain(tmp_path):
    runtime = FakePolicyRuntime()
    app = create_app(state_dir=tmp_path, deployment_runtime=runtime)
    client = TestClient(app)
    shard, manifest = fixture_shard("demo")
    shard_sha = hashlib.sha256(shard).hexdigest()
    create_body = {
        "object_key": "uploads/audit/demo.tar",
        "size_bytes": len(shard),
        "sha256": shard_sha,
    }
    upload = client.post(
        "/v1/uploads", headers={"Idempotency-Key": "audit-upload"}, json=create_body
    ).json()
    assert (
        client.post(
            "/v1/uploads", headers={"Idempotency-Key": "audit-upload"}, json=create_body
        ).json()["id"]
        == upload["id"]
    )
    assert client.put(upload["upload_url"], content=shard).json()["state"] == "uploaded"
    assert client.post(f"/v1/uploads/{upload['id']}/inspect").json()["actual_sha256"] == shard_sha
    commit_request = {"dataset_id": "audit-raw", "shard_id": "audit-shard", "manifest": manifest}
    receipt = client.post(f"/v1/uploads/{upload['id']}/commit", json=commit_request).json()
    assert client.post(f"/v1/uploads/{upload['id']}/commit", json=commit_request).json() == receipt
    assert client.get(f"/v1/commits/{receipt['commit_id']}").json() == receipt
    assert client.get("/v1/datasets").json()["datasets"][0]["episode_count"] == 1
    assert client.get("/v1/datasets/audit-raw/shards").json()["shards"][0]["sha256"] == shard_sha
    assert client.get("/v1/datasets/audit-raw/episodes").json()["episodes"][0]["id"] == "demo"

    snapshot = client.post(
        "/v1/datasets/audit-raw/snapshots", json={"control_source": "human"}
    ).json()
    assert (
        client.post("/v1/datasets/audit-raw/snapshots", json={"control_source": "human"}).json()[
            "snapshot_id"
        ]
        == snapshot["snapshot_id"]
    )
    with app.state.ingest.storage.open(upload["object_key"]) as source:
        assert [
            row["frame_index"] for row in iter_webdataset_rows(source, control_source="human")
        ] == [1]

    train_body = {
        "snapshot_id": snapshot["snapshot_id"],
        "config": {"algorithm": "bc", "epochs": 1},
    }
    job = client.post(
        "/v1/training/jobs", headers={"Idempotency-Key": "audit-bc"}, json=train_body
    ).json()
    assert (
        client.post(
            "/v1/training/jobs", headers={"Idempotency-Key": "audit-bc"}, json=train_body
        ).json()["job_id"]
        == job["job_id"]
    )
    assert job["state"] == "succeeded"
    assert client.get(f"/v1/training/jobs/{job['job_id']}/logs").json()["logs"]
    assert (
        client.get(f"/v1/training/jobs/{job['job_id']}/metrics").json()["metrics"]["loss"] == 0.125
    )

    def register(name, **compat):
        body = {
            "model_id": "audit-bc",
            "checkpoint_path": f"fake://{name}.pt",
            "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "source_snapshot_id": snapshot["snapshot_id"],
            "source_config": train_body["config"],
            "source_commit_id": receipt["commit_id"],
            "compatibility": {"action_spec_id": "switch_packets.v1", "action_dim": 26, **compat},
            "evaluation": {"loss": 0.125},
        }
        return client.post("/v1/models/revisions", json=body).json()

    one = register("one")
    client.post(f"/v1/models/revisions/{one['revision_id']}/validate")
    promote_one = client.post(
        f"/v1/models/revisions/{one['revision_id']}/promote",
        headers={"Idempotency-Key": "promote-one"},
        json={"expected_revision": None},
    ).json()
    assert (
        client.post(
            f"/v1/models/revisions/{one['revision_id']}/promote",
            headers={"Idempotency-Key": "promote-one"},
            json={"expected_revision": None},
        ).json()
        == promote_one
    )
    bad = register("bad", smoke_fail=True)
    assert client.post(f"/v1/models/revisions/{bad['revision_id']}/validate").status_code == 422
    assert client.get("/v1/deployment").json()["active_revision"] == one["revision_id"]
    two = register("two")
    client.post(f"/v1/models/revisions/{two['revision_id']}/validate")
    client.post(
        f"/v1/models/revisions/{two['revision_id']}/promote",
        headers={"Idempotency-Key": "promote-two"},
        json={"expected_revision": one["revision_id"]},
    )
    state = client.get("/v1/deployment").json()
    assert (
        state["active_revision"] == two["revision_id"]
        and state["previous_revision"] == one["revision_id"]
    )
    rolled = client.post(
        "/v1/deployment/rollback",
        headers={"Idempotency-Key": "rollback-one"},
        json={"expected_revision": two["revision_id"]},
    ).json()
    assert (
        rolled["active_revision"] == one["revision_id"]
        and runtime.active_revision == one["revision_id"]
    )

    spec = client.get("/openapi.json").json()
    required = {
        "/v1/uploads",
        "/v1/uploads/{upload_id}/content",
        "/v1/uploads/{upload_id}/inspect",
        "/v1/uploads/{upload_id}/commit",
        "/v1/commits/{commit_id}",
        "/v1/datasets",
        "/v1/datasets/{dataset_id}/shards",
        "/v1/datasets/{dataset_id}/episodes",
        "/v1/datasets/{dataset_id}/snapshots",
        "/v1/snapshots/{snapshot_id}",
        "/v1/training/jobs",
        "/v1/training/jobs/{job_id}",
        "/v1/training/jobs/{job_id}/logs",
        "/v1/training/jobs/{job_id}/metrics",
        "/v1/models/revisions",
        "/v1/models/revisions/{revision_id}",
        "/v1/models/revisions/{revision_id}/validate",
        "/v1/models/revisions/{revision_id}/promote",
        "/v1/deployment",
        "/v1/deployment/rollback",
    }
    assert required <= set(spec["paths"])
    assert {
        "CreateUploadRequest",
        "CommitRequest",
        "SnapshotRequest",
        "TrainingRequest",
        "RegisterModelRequest",
        "ActivationRequest",
    } <= set(spec["components"]["schemas"])


def test_openapi_success_responses_are_typed(tmp_path):
    spec = TestClient(create_app(state_dir=tmp_path)).get("/openapi.json").json()
    operations = [
        operation
        for item in spec["paths"].values()
        for operation in item.values()
        if isinstance(operation, dict) and "responses" in operation
    ]
    for operation in operations:
        success = operation["responses"].get("200") or operation["responses"].get("201")
        if success and "content" in success:
            for media in success["content"].values():
                assert media["schema"], operation["operationId"]
