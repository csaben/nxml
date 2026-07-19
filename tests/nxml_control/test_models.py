import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from nxml_control.models import CandidateError, ConflictError, FakePolicyRuntime, ModelRegistry


def register(registry, name="m", **compat):
    info = {"action_spec_id": "switch_packets.v1", "action_dim": 26, **compat}
    return registry.register(
        model_id=name,
        checkpoint_path=f"fake://{name}.pt",
        checkpoint_sha256=hashlib.sha256(name.encode()).hexdigest(),
        source_snapshot_id="sha256:" + "a" * 64,
        source_config={"epochs": 1},
        source_commit_id="commit-1",
        compatibility=info,
        evaluation={"loss": 0.1},
    )


def test_candidate_validate_promote_retry_and_rollback(tmp_path):
    runtime = FakePolicyRuntime()
    registry = ModelRegistry(tmp_path / "models.sqlite3", runtime)
    one = register(registry, "one")
    registry.validate(one["revision_id"])
    first = registry.promote(one["revision_id"], expected_revision=None, idempotency_key="p1")
    assert (
        registry.promote(one["revision_id"], expected_revision=None, idempotency_key="p1") == first
    )
    two = register(registry, "two")
    registry.validate(two["revision_id"])
    second = registry.promote(
        two["revision_id"], expected_revision=one["revision_id"], idempotency_key="p2"
    )
    assert second["previous_revision"] == one["revision_id"]
    rolled = registry.rollback(expected_revision=two["revision_id"], idempotency_key="r1")
    assert (
        rolled["active_revision"] == one["revision_id"]
        and runtime.active_revision == one["revision_id"]
    )


def test_candidate_failure_leaves_active_unchanged(tmp_path):
    runtime = FakePolicyRuntime()
    registry = ModelRegistry(tmp_path / "models.sqlite3", runtime)
    good = register(registry, "good")
    registry.validate(good["revision_id"])
    registry.promote(good["revision_id"], expected_revision=None, idempotency_key="good")
    bad = register(registry, "bad", smoke_fail=True)
    with pytest.raises(CandidateError):
        registry.validate(bad["revision_id"])
    assert (
        registry.deployment()["active_revision"] == good["revision_id"]
        and registry.get(bad["revision_id"])["state"] == "rejected"
    )


def test_concurrent_activation_has_single_cas_winner(tmp_path):
    registry = ModelRegistry(tmp_path / "models.sqlite3", FakePolicyRuntime())
    base = register(registry, "base")
    registry.validate(base["revision_id"])
    registry.promote(base["revision_id"], expected_revision=None, idempotency_key="base")
    candidates = [register(registry, name) for name in ("a", "b")]
    for item in candidates:
        registry.validate(item["revision_id"])

    def activate(item):
        try:
            return registry.promote(
                item["revision_id"],
                expected_revision=base["revision_id"],
                idempotency_key=item["revision_id"],
            )
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(activate, candidates))
    assert sum(result is not None for result in results) == 1


def test_runtime_activation_failure_and_failed_rollback_preserve_active(tmp_path):
    runtime = FakePolicyRuntime()
    registry = ModelRegistry(tmp_path / "models.sqlite3", runtime)
    one = register(registry, "one")
    registry.validate(one["revision_id"])
    registry.promote(one["revision_id"], expected_revision=None, idempotency_key="one")
    two = register(registry, "two")
    registry.validate(two["revision_id"])
    runtime.fail_activate.add(two["revision_id"])
    with pytest.raises(RuntimeError, match="runtime activation failed"):
        registry.promote(
            two["revision_id"], expected_revision=one["revision_id"], idempotency_key="two"
        )
    assert registry.deployment()["active_revision"] == one["revision_id"]
    runtime.fail_activate.clear()
    registry.promote(
        two["revision_id"], expected_revision=one["revision_id"], idempotency_key="two-retry"
    )
    runtime.fail_activate.add(one["revision_id"])
    with pytest.raises(RuntimeError, match="runtime activation failed"):
        registry.rollback(expected_revision=two["revision_id"], idempotency_key="rollback")
    assert registry.deployment()["active_revision"] == two["revision_id"]


def test_policy_server_runtime_preloads_smokes_then_swaps():
    import numpy as np
    from nxml_control.models import PolicyServerRuntime

    class Info:
        sequence_length = 2
        latent_shape = (1, 2, 2)
        action_dim = 26

        def __iter__(self):
            return iter({"sequence_length": 2, "latent_shape": (1, 2, 2), "action_dim": 26}.items())

    class Server:
        def __init__(self, model_path, device):
            self.path = model_path

        def info(self):
            return dict(Info())

        def predict(self, x):
            assert x.shape == (2, 1, 2, 2)
            return np.zeros(26, dtype=np.float32)

    runtime = PolicyServerRuntime(device="cpu", server_factory=Server)
    revision = {
        "revision_id": "r1",
        "checkpoint_path": "fake.pt",
        "compatibility": {"action_spec_id": "switch_packets.v1"},
    }
    candidate = runtime.prepare(revision)
    runtime.smoke(candidate)
    runtime.activate(candidate)
    assert runtime.server.path == "fake.pt"


def test_model_api_response_loss_retry_and_conflict(tmp_path):
    from fastapi.testclient import TestClient
    from nxml_control.api import create_app

    client = TestClient(create_app(state_dir=tmp_path))
    body = {
        "model_id": "bc",
        "checkpoint_path": "fake://bc.pt",
        "checkpoint_sha256": "a" * 64,
        "source_snapshot_id": "sha256:" + "b" * 64,
        "source_config": {},
        "source_commit_id": "c",
        "compatibility": {"action_spec_id": "switch_packets.v1", "action_dim": 26},
        "evaluation": {"loss": 0.1},
    }
    revision = client.post("/v1/models/revisions", json=body).json()
    rid = revision["revision_id"]
    assert client.post(f"/v1/models/revisions/{rid}/validate").json()["state"] == "validated"
    headers = {"Idempotency-Key": "promote-api"}
    request = {"expected_revision": None}
    first = client.post(f"/v1/models/revisions/{rid}/promote", headers=headers, json=request)
    assert first.status_code == 200
    assert (
        client.post(f"/v1/models/revisions/{rid}/promote", headers=headers, json=request).json()
        == first.json()
    )
    other = client.post(
        "/v1/models/revisions", json={**body, "checkpoint_path": "fake://other.pt"}
    ).json()
    client.post(f"/v1/models/revisions/{other['revision_id']}/validate")
    conflict = client.post(
        f"/v1/models/revisions/{other['revision_id']}/promote",
        headers={"Idempotency-Key": "stale"},
        json={"expected_revision": None},
    )
    assert conflict.status_code == 409
    assert client.get("/v1/deployment").json()["active_revision"] == rid
