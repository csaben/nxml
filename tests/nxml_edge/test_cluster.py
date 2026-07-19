from __future__ import annotations

import pytest
from nxml_edge.cluster import ClusterClient, ClusterDashboard, ClusterError


class FakeControl:
    def __init__(self):
        self.outage = False
        self.loss_once = False
        self.calls = []
        self.deployment = {
            "active_revision": "rev-a",
            "previous_revision": "rev-old",
            "generation": 4,
        }

    def __call__(self, method, path, body, key):
        self.calls.append((method, path, body, key))
        if self.outage:
            raise ClusterError("tailnet unavailable")
        if path == "/healthz":
            return {"status": "ok", "created": 3, "uploaded": 2, "committed": 1}
        if path == "/v1/datasets":
            return {"datasets": [{"id": "ds", "shard_count": 2, "episode_count": 7}]}
        if path == "/v1/snapshots":
            return {"snapshots": []}
        if path == "/v1/training/jobs" and method == "GET":
            return {"jobs": []}
        if path == "/v1/models/revisions":
            return {"revisions": []}
        if path == "/v1/deployments":
            return {"deployments": [dict(self.deployment)]}
        if path.endswith("/snapshots"):
            return {
                "snapshot_id": "sha256:" + "a" * 64,
                "dataset_id": "ds",
                "created_at": "now",
                "control_source": "human",
                "manifest": {},
            }
        if path == "/v1/training/jobs":
            response = {
                "job_id": "job",
                "idempotency_key": key,
                "snapshot_id": body["snapshot_id"],
                "config": body["config"],
                "state": "succeeded",
                "created_at": "now",
                "updated_at": "now",
            }
            if self.loss_once:
                self.loss_once = False
                raise ClusterError("response lost")
            return response
        if path.endswith("/promote"):
            if body["expected_generation"] != self.deployment["generation"]:
                raise ClusterError("generation conflict", status=409)
            if path.endswith("rev-bad/promote"):
                raise ClusterError("activation failed", status=422)
            self.deployment = {
                "active_revision": path.split("/")[-2],
                "previous_revision": "rev-a",
                "generation": 5,
            }
            return {"operation": "promote", **self.deployment}
        if path == "/v1/deployment/rollback":
            self.deployment = {
                "active_revision": "rev-old",
                "previous_revision": "rev-a",
                "generation": 5,
            }
            return {"operation": "rollback", **self.deployment}
        raise AssertionError(path)


def test_cluster_outage_returns_stale_cache_without_touching_session(edge):
    transport = FakeControl()
    dashboard = ClusterDashboard(ClusterClient(transport))
    assert dashboard.status()["episode_count"] == 7
    transport.outage = True
    assert dashboard.status()["stale"] is True
    assert edge[0].status().ready is True


def test_response_loss_retry_reuses_idempotency_key():
    transport = FakeControl()
    transport.loss_once = True
    job = ClusterClient(transport).submit_bc("sha256:" + "a" * 64, {}, "same-key")
    assert job.job_id == "job"
    calls = [call for call in transport.calls if call[1] == "/v1/training/jobs"]
    assert [call[3] for call in calls] == ["same-key", "same-key"]


def test_promotion_generation_cas_conflict():
    with pytest.raises(ClusterError) as error:
        ClusterClient(FakeControl()).promote("rev-b", 3, "key")
    assert error.value.status == 409


def test_failed_activation_preserves_active_revision():
    transport = FakeControl()
    with pytest.raises(ClusterError):
        ClusterClient(transport).promote("rev-bad", 4, "key")
    assert transport.deployment["active_revision"] == "rev-a"
    assert transport.deployment["generation"] == 4


def test_rollback_uses_generation_cas_and_returns_previous():
    transport = FakeControl()
    result = ClusterClient(transport).rollback(4, "rollback-key")
    assert result.active_revision == "rev-old"
    assert transport.calls[-1][2] == {"expected_generation": 4}


def test_human_snapshot_is_explicitly_filtered():
    transport = FakeControl()
    snapshot = ClusterClient(transport).snapshot("ds", "snapshot-key")
    assert snapshot.control_source == "human"
    assert transport.calls[-1][2] == {"control_source": "human"}
    assert transport.calls[-1][3] == "snapshot-key"
