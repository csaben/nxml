import pytest
from nxml_control.training import FakeTrainingExecutor, TrainingJobs, TrainingSpec

from tests.nxml_control.fixture_shard import fixture_shard


def test_fake_bc_job_persists_status_logs_metrics_and_lineage(tmp_path):
    jobs = TrainingJobs(tmp_path / "catalog.sqlite3", FakeTrainingExecutor())
    spec = TrainingSpec("sha256:snapshot", {"epochs": 1, "control_source": "human"}, "train-1")
    job = jobs.submit(spec)
    assert job["state"] == "succeeded" and job["snapshot_id"] == spec.snapshot_id
    assert len(job["checkpoint_sha256"]) == 64
    assert jobs.logs(job["job_id"])[-1] == "bc smoke complete"
    assert jobs.metrics(job["job_id"])["loss"] == 0.125
    assert jobs.submit(spec)["job_id"] == job["job_id"]


def test_training_idempotency_conflict(tmp_path):
    jobs = TrainingJobs(tmp_path / "catalog.sqlite3", FakeTrainingExecutor())
    jobs.submit(TrainingSpec("s1", {}, "same"))
    with pytest.raises(ValueError, match="different training request"):
        jobs.submit(TrainingSpec("s2", {}, "same"))


class FailingExecutor:
    def execute(self, job_id, spec):
        raise RuntimeError("GPU worker rejected job")


def test_executor_failure_is_persisted(tmp_path):
    jobs = TrainingJobs(tmp_path / "catalog.sqlite3", FailingExecutor())
    job = jobs.submit(TrainingSpec("s", {}, "fail"))
    assert job["state"] == "failed" and job["error"] == "GPU worker rejected job"


def test_training_http_uses_immutable_snapshot(tmp_path):
    import hashlib

    from fastapi.testclient import TestClient
    from nxml_control.api import create_app

    client = TestClient(create_app(state_dir=tmp_path))
    content, manifest = fixture_shard("e")
    digest = hashlib.sha256(content).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "training-shard"},
        json={"object_key": "uploads/training.tar", "size_bytes": len(content), "sha256": digest},
    ).json()
    client.put(upload["upload_url"], content=content)
    client.post(
        f"/v1/uploads/{upload['id']}/commit",
        json={"dataset_id": "raw", "shard_id": "s", "manifest": manifest},
    )
    snapshot = client.post("/v1/datasets/raw/snapshots", json={"control_source": "human"}).json()
    response = client.post(
        "/v1/training/jobs",
        headers={"Idempotency-Key": "bc-1"},
        json={"snapshot_id": snapshot["snapshot_id"], "config": {"epochs": 1}},
    )
    assert response.status_code == 201
    job = response.json()
    assert job["state"] == "succeeded"
    assert (
        client.get(f"/v1/training/jobs/{job['job_id']}/logs").json()["logs"][-1]
        == "bc smoke complete"
    )
    assert (
        client.get(f"/v1/training/jobs/{job['job_id']}/metrics").json()["metrics"]["loss"] == 0.125
    )
