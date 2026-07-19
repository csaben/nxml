import hashlib

from fastapi.testclient import TestClient
from nxml_control.api import create_app

from tests.nxml_control.fixture_shard import fixture_shard


def seed(client):
    shard, manifest = fixture_shard("e")
    digest = hashlib.sha256(shard).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "lists-upload"},
        json={"object_key": "uploads/lists.tar", "size_bytes": len(shard), "sha256": digest},
    ).json()
    client.put(upload["upload_url"], content=shard)
    receipt = client.post(
        f"/v1/uploads/{upload['id']}/commit",
        json={"dataset_id": "raw", "shard_id": "s", "manifest": manifest},
    ).json()
    human = client.post("/v1/datasets/raw/snapshots", json={"control_source": "human"}).json()
    policy = client.post("/v1/datasets/raw/snapshots", json={"control_source": "policy"}).json()
    return receipt, human, policy


def model_body(name, snapshot, commit, state_hint=None):
    return {
        "model_id": "bc",
        "checkpoint_path": f"fake://{name}.pt",
        "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "source_snapshot_id": snapshot,
        "source_config": {},
        "source_commit_id": commit,
        "compatibility": {"action_spec_id": "switch_packets.v1", "action_dim": 26},
        "evaluation": {},
    }


def test_deterministic_bounded_filtered_lists(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    receipt, human, policy = seed(client)
    snapshots = client.get("/v1/snapshots", params={"dataset_id": "raw", "limit": 1}).json()[
        "snapshots"
    ]
    assert len(snapshots) == 1
    assert (
        client.get("/v1/snapshots", params={"control_source": "human"}).json()["snapshots"][0][
            "snapshot_id"
        ]
        == human["snapshot_id"]
    )
    assert client.get("/v1/snapshots").json() == client.get("/v1/snapshots").json()

    for key, snapshot in (("j1", human), ("j2", policy)):
        client.post(
            "/v1/training/jobs",
            headers={"Idempotency-Key": key},
            json={"snapshot_id": snapshot["snapshot_id"], "config": {"epochs": 1}},
        )
    jobs = client.get(
        "/v1/training/jobs", params={"state": "succeeded", "snapshot_id": human["snapshot_id"]}
    ).json()["jobs"]
    assert [job["idempotency_key"] for job in jobs] == ["j1"]
    assert (
        client.get("/v1/training/jobs", params={"limit": 1, "offset": 1}).json()
        == client.get("/v1/training/jobs", params={"limit": 1, "offset": 1}).json()
    )

    one = client.post(
        "/v1/models/revisions", json=model_body("one", human["snapshot_id"], receipt["commit_id"])
    ).json()
    two = client.post(
        "/v1/models/revisions", json=model_body("two", human["snapshot_id"], receipt["commit_id"])
    ).json()
    client.post(f"/v1/models/revisions/{one['revision_id']}/validate")
    candidates = client.get(
        "/v1/models/revisions", params={"model_id": "bc", "state": "candidate"}
    ).json()["revisions"]
    assert [item["revision_id"] for item in candidates] == [two["revision_id"]]
    assert (
        client.get("/v1/models/revisions", params={"limit": 1}).json()
        == client.get("/v1/models/revisions", params={"limit": 1}).json()
    )
    assert client.get("/v1/deployments").json() == {
        "deployments": [{"active_revision": None, "previous_revision": None, "generation": 0}]
    }


def test_generation_cas_stale_retry_and_mutation_payloads(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    receipt, human, _ = seed(client)
    one = client.post(
        "/v1/models/revisions", json=model_body("one", human["snapshot_id"], receipt["commit_id"])
    ).json()
    two = client.post(
        "/v1/models/revisions", json=model_body("two", human["snapshot_id"], receipt["commit_id"])
    ).json()
    for item in (one, two):
        client.post(f"/v1/models/revisions/{item['revision_id']}/validate")
    headers = {"Idempotency-Key": "generation-one"}
    request = {"expected_generation": 0}
    first = client.post(
        f"/v1/models/revisions/{one['revision_id']}/promote", headers=headers, json=request
    ).json()
    assert first == {
        "operation": "promote",
        "active_revision": one["revision_id"],
        "previous_revision": None,
        "generation": 1,
    }
    assert (
        client.post(
            f"/v1/models/revisions/{one['revision_id']}/promote", headers=headers, json=request
        ).json()
        == first
    )
    stale = client.post(
        f"/v1/models/revisions/{two['revision_id']}/promote",
        headers={"Idempotency-Key": "stale-generation"},
        json={"expected_generation": 0},
    )
    assert stale.status_code == 409
    second = client.post(
        f"/v1/models/revisions/{two['revision_id']}/promote",
        headers={"Idempotency-Key": "generation-two"},
        json={"expected_generation": 1},
    ).json()
    assert second["generation"] == 2 and second["previous_revision"] == one["revision_id"]
    assert (
        client.post(
            "/v1/deployment/rollback",
            headers={"Idempotency-Key": "stale-rollback"},
            json={"expected_generation": 1},
        ).status_code
        == 409
    )
    rolled = client.post(
        "/v1/deployment/rollback",
        headers={"Idempotency-Key": "generation-rollback"},
        json={"expected_generation": 2},
    ).json()
    assert (
        rolled["generation"] == 3
        and rolled["active_revision"] == one["revision_id"]
        and rolled["previous_revision"] == two["revision_id"]
    )
