"""Typed, failure-isolated client for the ml-stream control plane."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ClusterError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class Health(BaseModel):
    """Public control-plane readiness response.

    Current servers return only ``{"status": "ready"}``.  The optional
    counters retain wire compatibility with older servers, but dashboard
    counts must come from authenticated list endpoints rather than health.
    """

    status: Literal["ready", "ok"]
    created: int | None = None
    uploaded: int | None = None
    committed: int | None = None


class Dataset(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    shard_count: int = 0
    episode_count: int = 0


class Snapshot(BaseModel):
    snapshot_id: str
    dataset_id: str
    created_at: str
    control_source: Literal["all", "human", "policy"]
    manifest: dict[str, Any]


class TrainingJob(BaseModel):
    job_id: str
    idempotency_key: str
    snapshot_id: str
    config: dict[str, Any]
    state: str
    created_at: str
    updated_at: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelRevision(BaseModel):
    revision_id: str
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_snapshot_id: str
    source_config: dict[str, Any]
    source_commit_id: str
    compatibility: dict[str, Any]
    evaluation: dict[str, Any]
    state: str
    created_at: str


class Deployment(BaseModel):
    active_revision: str | None
    previous_revision: str | None
    generation: int


class Activation(Deployment):
    operation: Literal["promote", "rollback"]


class Receipt(BaseModel):
    commit_id: str
    upload_id: str
    checksum: str
    size_bytes: int
    storage_key: str
    state: str
    committed_at: str
    dataset_id: str
    shard_id: str


# Search is deliberately only a client/view seam. No vector or embedding work runs here.
class ClipRef(BaseModel):
    clip_id: str
    episode_id: str
    start_frame: int
    end_frame: int
    thumbnail_url: str | None = None


class ClipLabel(BaseModel):
    clip_id: str
    label: str
    source: Literal["human", "system"] = "human"


class SearchQuery(BaseModel):
    text: str
    game: str | None = None
    policy_revision: str | None = None
    driver: str | None = None


@dataclass
class HttpTransport:
    base_url: str
    token: str | None = None
    timeout: float = 5.0

    def __call__(self, method: str, path: str, body: dict | None, key: str | None) -> Any:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if key:
            headers["Idempotency-Key"] = key
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise ClusterError(detail or str(error), status=error.code) from error
        except (OSError, TimeoutError) as error:
            raise ClusterError(str(error)) from error


class ClusterClient:
    """Strict models plus response-loss-safe idempotent mutation retries."""

    def __init__(self, transport: Callable[[str, str, dict | None, str | None], Any]) -> None:
        self._request = transport

    def health(self) -> Health:
        return Health.model_validate(self._request("GET", "/healthz", None, None))

    def datasets(self) -> list[Dataset]:
        data = self._request("GET", "/v1/datasets", None, None)
        return [Dataset.model_validate(item) for item in data["datasets"]]

    def snapshots(self) -> list[Snapshot]:
        data = self._request("GET", "/v1/snapshots", None, None)
        return [Snapshot.model_validate(item) for item in data["snapshots"]]

    def jobs(self) -> list[TrainingJob]:
        data = self._request("GET", "/v1/training/jobs", None, None)
        return [TrainingJob.model_validate(item) for item in data["jobs"]]

    def revisions(self) -> list[ModelRevision]:
        data = self._request("GET", "/v1/models/revisions", None, None)
        return [ModelRevision.model_validate(item) for item in data["revisions"]]

    def deployments(self) -> list[Deployment]:
        data = self._request("GET", "/v1/deployments", None, None)
        return [Deployment.model_validate(item) for item in data["deployments"]]

    def snapshot(self, dataset_id: str, key: str) -> Snapshot:
        return Snapshot.model_validate(
            self._retry(
                "POST", f"/v1/datasets/{dataset_id}/snapshots", {"control_source": "human"}, key
            )
        )

    def submit_bc(self, snapshot_id: str, config: dict, key: str) -> TrainingJob:
        data = self._retry(
            "POST", "/v1/training/jobs", {"snapshot_id": snapshot_id, "config": config}, key
        )
        return TrainingJob.model_validate(data)

    def job(self, job_id: str) -> TrainingJob:
        job = TrainingJob.model_validate(
            self._request("GET", f"/v1/training/jobs/{job_id}", None, None)
        )
        job.logs = self._request("GET", f"/v1/training/jobs/{job_id}/logs", None, None)["logs"]
        job.metrics = self._request("GET", f"/v1/training/jobs/{job_id}/metrics", None, None)[
            "metrics"
        ]
        return job

    def revision(self, revision_id: str) -> ModelRevision:
        return ModelRevision.model_validate(
            self._request("GET", f"/v1/models/revisions/{revision_id}", None, None)
        )

    def receipt(self, commit_id: str) -> Receipt:
        return Receipt.model_validate(self._request("GET", f"/v1/commits/{commit_id}", None, None))

    def promote(self, revision_id: str, expected_generation: int, key: str):
        return Activation.model_validate(
            self._retry(
                "POST",
                f"/v1/models/revisions/{revision_id}/promote",
                {"expected_generation": expected_generation},
                key,
            )
        )

    def rollback(self, expected_generation: int, key: str):
        return Activation.model_validate(
            self._retry(
                "POST", "/v1/deployment/rollback", {"expected_generation": expected_generation}, key
            )
        )

    def _retry(self, method: str, path: str, body: dict, key: str):
        try:
            return self._request(method, path, body, key)
        except ClusterError as error:
            if error.status is not None:
                raise
            return self._request(method, path, body, key)


class ClusterDashboard:
    """Cluster cache isolated from all capture/session-control readiness."""

    def __init__(self, client: ClusterClient, *, stale_after: float = 15.0) -> None:
        self.client = client
        self.stale_after = stale_after
        self.snapshots: dict[str, Snapshot] = {}
        self.jobs: dict[str, TrainingJob] = {}
        self.revisions: dict[str, ModelRevision] = {}
        self.receipts: dict[str, Receipt] = {}
        self._last_good: dict[str, Any] | None = None
        self._last_good_at: float | None = None
        self.error: str | None = None

    def status(self) -> dict[str, Any]:
        try:
            health = self.client.health()
            datasets = self.client.datasets()
            self.snapshots = {item.snapshot_id: item for item in self.client.snapshots()}
            self.jobs = {item.job_id: item for item in self.client.jobs()}
            self.revisions = {item.revision_id: item for item in self.client.revisions()}
            deployments = self.client.deployments()
            deployment = (
                deployments[0]
                if deployments
                else Deployment(active_revision=None, previous_revision=None, generation=0)
            )
            for job_id in list(self.jobs):
                self.jobs[job_id] = self.client.job(job_id)
            result = self._view(health, datasets, deployment)
            self._last_good, self._last_good_at, self.error = result, time.monotonic(), None
            return result
        except (ClusterError, ValueError, KeyError) as error:
            age = None if self._last_good_at is None else time.monotonic() - self._last_good_at
            result = dict(self._last_good or self._empty())
            result.update(connected=False, error=str(error), stale=True, stale_age_seconds=age)
            self.error = str(error)
            return result

    def create_snapshot(self, dataset_id: str, key: str | None = None) -> Snapshot:
        item = self.client.snapshot(dataset_id, key or str(uuid4()))
        self.snapshots[item.snapshot_id] = item
        return item

    def submit_bc(self, snapshot_id: str, config: dict, key: str | None = None) -> TrainingJob:
        item = self.client.submit_bc(snapshot_id, config, key or str(uuid4()))
        self.jobs[item.job_id] = item
        return item

    def promote(self, revision_id: str, generation: int, key: str | None = None):
        return self.client.promote(revision_id, generation, key or str(uuid4()))

    def rollback(self, generation: int, key: str | None = None):
        return self.client.rollback(generation, key or str(uuid4()))

    def _view(self, health, datasets, deployment):
        now = time.monotonic()
        return {
            "connected": True,
            "error": None,
            "stale": False,
            "stale_age_seconds": 0.0,
            "health": health,
            "datasets": datasets,
            "dataset_count": len(datasets),
            "shard_count": sum(x.shard_count for x in datasets),
            "episode_count": sum(x.episode_count for x in datasets),
            "snapshots": list(self.snapshots.values()),
            "jobs": list(self.jobs.values()),
            "revisions": list(self.revisions.values()),
            "receipts": list(self.receipts.values()),
            "deployment": deployment,
            "updated_monotonic": now,
        }

    @staticmethod
    def _empty():
        return {
            "connected": False,
            "health": None,
            "datasets": [],
            "dataset_count": 0,
            "shard_count": 0,
            "episode_count": 0,
            "snapshots": [],
            "jobs": [],
            "revisions": [],
            "receipts": [],
            "deployment": None,
        }
