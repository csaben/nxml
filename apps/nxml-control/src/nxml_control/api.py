from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from nxml_control.catalog import Catalog, IdentityConflictError, InvalidManifestError, Upload
from nxml_control.models import (
    CandidateError,
    ConflictError,
    DeploymentRuntime,
    FakePolicyRuntime,
    ModelRegistry,
)
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage
from nxml_control.training import FakeTrainingExecutor, TrainingExecutor, TrainingJobs, TrainingSpec


class HealthResponse(BaseModel):
    status: str
    created: int
    uploaded: int
    committed: int


class UploadResponse(BaseModel):
    id: str
    idempotency_key: str
    object_key: str
    expected_size_bytes: int
    expected_sha256: str
    state: str
    actual_size_bytes: int | None
    actual_sha256: str | None
    dataset_id: str | None
    shard_id: str | None
    upload_url: str


class CommitReceiptResponse(BaseModel):
    commit_id: str
    upload_id: str
    checksum: str
    size_bytes: int
    storage_key: str
    state: str
    committed_at: str
    dataset_id: str
    shard_id: str


class SnapshotResponse(BaseModel):
    snapshot_id: str
    dataset_id: str
    created_at: str
    control_source: str
    manifest: dict[str, Any]


class TrainingJobResponse(BaseModel):
    job_id: str
    idempotency_key: str
    snapshot_id: str
    config: dict[str, Any]
    state: str
    created_at: str
    updated_at: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    error: str | None


class ModelRevisionResponse(BaseModel):
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


class DeploymentResponse(BaseModel):
    active_revision: str | None
    previous_revision: str | None
    generation: int


class ActivationResponse(BaseModel):
    operation: str
    active_revision: str
    previous_revision: str | None
    generation: int


class DatasetListResponse(BaseModel):
    datasets: list[dict[str, Any]]


class ShardListResponse(BaseModel):
    shards: list[dict[str, Any]]


class EpisodeListResponse(BaseModel):
    episodes: list[dict[str, Any]]


class LogsResponse(BaseModel):
    logs: list[str]


class MetricsResponse(BaseModel):
    metrics: dict[str, float]


class CreateUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_key: str = Field(pattern=r"^uploads/[A-Za-z0-9._/-]+\.tar$")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RegisterModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_id: str
    source_config: dict[str, Any]
    source_commit_id: str
    compatibility: dict[str, Any]
    evaluation: dict[str, Any]


class ActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: str | None = None


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config: dict[str, Any]


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control_source: str = Field(default="all", pattern=r"^(all|human|policy)$")


class CommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(min_length=1)
    shard_id: str = Field(min_length=1)
    manifest: dict[str, Any]


def _view(item: Upload) -> dict[str, Any]:
    return {**item.__dict__, "upload_url": f"/v1/uploads/{item.id}/content"}


def create_app(
    *,
    state_dir: str | Path,
    training_executor: TrainingExecutor | None = None,
    deployment_runtime: DeploymentRuntime | None = None,
) -> FastAPI:
    state = Path(state_dir)
    catalog = Catalog(state / "catalog.sqlite3")
    service = IngestService(catalog, LocalObjectStorage(state / "objects"))
    training = TrainingJobs(state / "catalog.sqlite3", training_executor or FakeTrainingExecutor())
    models = ModelRegistry(state / "catalog.sqlite3", deployment_runtime or FakePolicyRuntime())
    app = FastAPI(title="NXML ML Control Plane", version="1.0.0")

    @app.get("/healthz", response_model=HealthResponse)
    def health():
        return catalog.health()

    @app.post("/v1/uploads", status_code=status.HTTP_201_CREATED, response_model=UploadResponse)
    def create(
        body: CreateUploadRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    ):
        try:
            return _view(
                catalog.create_upload(
                    idempotency_key=idempotency_key,
                    object_key=body.object_key,
                    size_bytes=body.size_bytes,
                    sha256=body.sha256,
                )
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.put("/v1/uploads/{upload_id}/content", response_model=UploadResponse)
    async def content(upload_id: str, request: Request):
        try:
            return _view(service.upload(upload_id, BytesIO(await request.body())))
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/uploads/{upload_id}/inspect", response_model=UploadResponse)
    def inspect(upload_id: str):
        try:
            return _view(service.inspect(upload_id))
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/uploads/{upload_id}/commit", response_model=CommitReceiptResponse)
    def commit(upload_id: str, body: CommitRequest):
        try:
            service.verify_members(upload_id, body.manifest)
            return catalog.commit(upload_id, **body.model_dump()).__dict__
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except InvalidManifestError as error:
            raise HTTPException(422, str(error)) from error
        except IdentityConflictError as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/datasets", response_model=DatasetListResponse)
    def datasets():
        return {"datasets": catalog.list_datasets()}

    @app.get("/v1/datasets/{dataset_id}/shards", response_model=ShardListResponse)
    def shards(dataset_id: str):
        return {"shards": [item.__dict__ for item in catalog.list_shards(dataset_id)]}

    @app.get("/v1/datasets/{dataset_id}/episodes", response_model=EpisodeListResponse)
    def episodes(dataset_id: str):
        return {"episodes": [item.__dict__ for item in catalog.list_episodes(dataset_id)]}

    @app.get("/v1/commits/{commit_id}", response_model=CommitReceiptResponse)
    def receipt(commit_id: str):
        try:
            return catalog.get_receipt(commit_id).__dict__
        except KeyError as error:
            raise HTTPException(404, "commit not found") from error

    @app.post(
        "/v1/datasets/{dataset_id}/snapshots", status_code=201, response_model=SnapshotResponse
    )
    def create_snapshot(dataset_id: str, body: SnapshotRequest):
        try:
            return catalog.create_snapshot(dataset_id, control_source=body.control_source).__dict__
        except KeyError as error:
            raise HTTPException(404, "dataset not found") from error

    @app.get("/v1/snapshots/{snapshot_id}", response_model=SnapshotResponse)
    def snapshot(snapshot_id: str):
        try:
            return catalog.get_snapshot(snapshot_id).__dict__
        except KeyError as error:
            raise HTTPException(404, "snapshot not found") from error

    @app.post("/v1/training/jobs", status_code=201, response_model=TrainingJobResponse)
    def submit_training(
        body: TrainingRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    ):
        try:
            catalog.get_snapshot(body.snapshot_id)
            return training.submit(TrainingSpec(body.snapshot_id, body.config, idempotency_key))
        except KeyError as error:
            raise HTTPException(404, "snapshot not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/v1/training/jobs/{job_id}", response_model=TrainingJobResponse)
    def training_job(job_id: str):
        try:
            return training.get(job_id)
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.get("/v1/training/jobs/{job_id}/logs", response_model=LogsResponse)
    def training_logs(job_id: str):
        try:
            return {"logs": training.logs(job_id)}
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.get("/v1/training/jobs/{job_id}/metrics", response_model=MetricsResponse)
    def training_metrics(job_id: str):
        try:
            return {"metrics": training.metrics(job_id)}
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.post("/v1/models/revisions", status_code=201, response_model=ModelRevisionResponse)
    def register_model(body: RegisterModelRequest):
        return models.register(**body.model_dump())

    @app.get("/v1/models/revisions/{revision_id}", response_model=ModelRevisionResponse)
    def model_revision(revision_id: str):
        try:
            return models.get(revision_id)
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error

    @app.post("/v1/models/revisions/{revision_id}/validate", response_model=ModelRevisionResponse)
    def validate_model(revision_id: str):
        try:
            return models.validate(revision_id)
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error
        except CandidateError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/deployment", response_model=DeploymentResponse)
    def deployment():
        return models.deployment()

    @app.post("/v1/models/revisions/{revision_id}/promote", response_model=ActivationResponse)
    def promote_model(
        revision_id: str,
        body: ActivationRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    ):
        try:
            return models.promote(
                revision_id,
                expected_revision=body.expected_revision,
                idempotency_key=idempotency_key,
            )
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error
        except CandidateError as error:
            raise HTTPException(422, str(error)) from error
        except ConflictError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/v1/deployment/rollback", response_model=ActivationResponse)
    def rollback_model(
        body: ActivationRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    ):
        try:
            return models.rollback(
                expected_revision=body.expected_revision, idempotency_key=idempotency_key
            )
        except CandidateError as error:
            raise HTTPException(422, str(error)) from error
        except ConflictError as error:
            raise HTTPException(409, str(error)) from error

    app.state.catalog = catalog
    app.state.ingest = service
    app.state.training = training
    app.state.models = models
    app.state.ingest = service
    return app
