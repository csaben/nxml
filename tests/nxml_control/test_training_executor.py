import hashlib
import json
import sys
import threading

from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.training import SubprocessTrainingExecutor, TrainingJobs, TrainingSpec


def test_subprocess_worker_manifest_and_verified_artifact(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--request", required=True)
p.add_argument("--result", required=True)
a = p.parse_args()
request = json.loads(Path(a.request).read_text())
checkpoint = Path(a.result).with_name("policy.pt")
checkpoint.write_bytes(b"verified-policy")
result = {
    "schema_id": "nxml.bc-result.v1",
    "job_id": request["job_id"],
    "snapshot_id": request["snapshot_id"],
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "metrics": {"loss": 0.25},
    "logs": ["worker complete"],
}
Path(a.result).write_text(json.dumps(result))
"""
    )
    executor = SubprocessTrainingExecutor([sys.executable, str(worker)], tmp_path / "jobs")
    jobs = TrainingJobs(tmp_path / "catalog.sqlite3", executor)
    job = jobs.submit(TrainingSpec("sha256:" + "1" * 64, {"epochs": 1}, "real-worker"))

    assert job["state"] == "succeeded"
    assert jobs.metrics(job["job_id"]) == {"loss": 0.25}
    artifact = jobs.artifacts(job["job_id"])[0]
    assert artifact["sha256"] == hashlib.sha256(b"verified-policy").hexdigest()
    job_dir = tmp_path / "jobs" / job["job_id"]
    request = json.loads((job_dir / "request.json").read_text())
    result = json.loads((job_dir / "result.json").read_text())
    assert request["schema_id"] == "nxml.bc-job.v1"
    assert result["schema_id"] == "nxml.bc-result.v1"
    assert (job_dir / "request.json").stat().st_mode & 0o777 == 0o440
    assert (job_dir / "result.json").stat().st_mode & 0o777 == 0o440


def test_subprocess_worker_rejects_bad_digest(tmp_path):
    worker = tmp_path / "bad_worker.py"
    worker.write_text(
        """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--request", required=True)
p.add_argument("--result", required=True)
a = p.parse_args()
request = json.loads(Path(a.request).read_text())
checkpoint = Path(a.result).with_name("policy.pt")
checkpoint.write_bytes(b"policy")
print("worker produced checkpoint", flush=True)
Path(a.result).write_text(json.dumps({
    "schema_id": "nxml.bc-result.v1",
    "job_id": request["job_id"],
    "snapshot_id": request["snapshot_id"],
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": "0" * 64,
}))
"""
    )
    jobs = TrainingJobs(
        tmp_path / "catalog.sqlite3",
        SubprocessTrainingExecutor([sys.executable, str(worker)], tmp_path / "jobs"),
    )
    job = jobs.submit(TrainingSpec("snapshot", {}, "bad-digest"))
    assert job["state"] == "failed"
    assert "digest mismatch" in job["error"]
    assert jobs.logs(job["job_id"]) == ["worker produced checkpoint"]


class BlockingExecutor:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, job_id, spec):
        self.started.set()
        self.release.wait(timeout=5)
        raise RuntimeError("cancelled")

    def cancel(self, job_id):
        self.release.set()
        return True


def test_async_queued_and_running_jobs_can_be_cancelled(tmp_path):
    executor = BlockingExecutor()
    jobs = TrainingJobs(tmp_path / "catalog.sqlite3", executor, run_async=True)
    first = jobs.submit(TrainingSpec("s", {}, "first"))
    assert executor.started.wait(timeout=2)
    second = jobs.submit(TrainingSpec("s", {}, "second"))

    assert jobs.cancel(second["job_id"])["state"] == "cancelled"
    assert jobs.cancel(first["job_id"])["state"] == "cancelled"


def test_api_artifacts_cancel_and_revision_lineage(tmp_path):
    from tests.nxml_control.test_training import test_training_http_uses_immutable_snapshot

    test_training_http_uses_immutable_snapshot(tmp_path)
    client = TestClient(create_app(state_dir=tmp_path))
    job = client.get("/v1/training/jobs").json()["jobs"][0]

    response = client.get(f"/v1/training/jobs/{job['job_id']}/artifacts")
    assert response.status_code == 200
    artifact = response.json()["artifacts"][0]
    body = {
        "model_id": "dagger",
        "checkpoint_path": artifact["uri"],
        "checkpoint_sha256": artifact["sha256"],
        "source_snapshot_id": job["snapshot_id"],
        "source_config": job["config"],
        "source_commit_id": "source-commit",
        "compatibility": {"action_spec_id": "switch_packets.v1", "action_dim": 26},
        "evaluation": {},
        "training_job_id": job["job_id"],
    }
    assert client.post("/v1/models/revisions", json=body).status_code == 201
    body["checkpoint_sha256"] = "0" * 64
    assert client.post("/v1/models/revisions", json=body).status_code == 422
    assert client.post(f"/v1/training/jobs/{job['job_id']}/cancel").status_code == 409

    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/training/jobs/{job_id}/cancel" in paths
    assert "/v1/training/jobs/{job_id}/artifacts" in paths


def test_production_can_disable_training_without_using_fake(tmp_path):
    from nxml_control.training import DisabledTrainingExecutor

    app = create_app(state_dir=tmp_path, training_executor=DisabledTrainingExecutor())
    assert app.state.training.available is False
    client = TestClient(app)
    response = client.post(
        "/v1/training/jobs",
        headers={"Idempotency-Key": "disabled"},
        json={"snapshot_id": "sha256:" + "0" * 64, "config": {}},
    )
    assert response.status_code == 503


def test_quality_disposition_excludes_episode_and_is_snapshot_lineage(tmp_path):
    from tests.nxml_control.fixture_shard import fixture_shard

    client = TestClient(create_app(state_dir=tmp_path))
    receipts = []
    for index, episode_id in enumerate(("bad", "good")):
        content, manifest = fixture_shard(episode_id)
        digest = hashlib.sha256(content).hexdigest()
        upload = client.post(
            "/v1/uploads",
            headers={"Idempotency-Key": f"quality-upload-{index}"},
            json={
                "object_key": f"uploads/quality-{index}.tar",
                "size_bytes": len(content),
                "sha256": digest,
            },
        ).json()
        assert client.put(upload["upload_url"], content=content).status_code == 200
        receipt = client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "quality", "shard_id": f"s-{index}", "manifest": manifest},
        )
        assert receipt.status_code == 200
        receipts.append(receipt.json())

    quality_body = {
        "schema_id": "nxml.episode-quality.v1",
        "training_eligible": False,
        "reason": "noncausal_action_alignment",
        "validator": "edge-action-alignment",
        "validator_version": "1",
    }
    endpoint = "/v1/datasets/quality/episodes/bad/quality-dispositions"
    first = client.post(
        endpoint, headers={"Idempotency-Key": "quarantine-bad-v1"}, json=quality_body
    )
    retry = client.post(
        endpoint, headers={"Idempotency-Key": "quarantine-bad-v1"}, json=quality_body
    )
    assert first.status_code == 201
    assert retry.json()["disposition_id"] == first.json()["disposition_id"]
    assert client.get(endpoint).json()["dispositions"] == [first.json()]
    listed_receipt = client.get(
        "/v1/commits", params={"dataset_id": "quality", "shard_id": "s-0"}
    ).json()["receipts"]
    assert listed_receipt == [receipts[0]]

    snapshot = client.post(
        "/v1/datasets/quality/snapshots", json={"control_source": "human"}
    ).json()
    assert snapshot["manifest"]["schema_id"] == "nxml.dataset-snapshot.v1"
    assert snapshot["manifest"]["shards"] == [
        {
            "id": "s-1",
            "sha256": receipts[1]["checksum"],
            "size_bytes": receipts[1]["size_bytes"],
            "object_key": receipts[1]["storage_key"],
            "episode_ids": ["good"],
        }
    ]
    assert snapshot["manifest"]["excluded_episodes"][0]["episode_id"] == "bad"
    assert snapshot["manifest"]["excluded_episodes"][0]["reason"] == "noncausal_action_alignment"
