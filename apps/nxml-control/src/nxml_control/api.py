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

    @app.get("/healthz")
    def health():
        return catalog.health()

    @app.post("/v1/uploads", status_code=status.HTTP_201_CREATED)
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

    @app.put("/v1/uploads/{upload_id}/content")
    async def content(upload_id: str, request: Request):
        try:
            return _view(service.upload(upload_id, BytesIO(await request.body())))
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/uploads/{upload_id}/inspect")
    def inspect(upload_id: str):
        try:
            return _view(service.inspect(upload_id))
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/uploads/{upload_id}/commit")
    def commit(upload_id: str, body: CommitRequest):
        try:
            return catalog.commit(upload_id, **body.model_dump()).__dict__
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except InvalidManifestError as error:
            raise HTTPException(422, str(error)) from error
        except IdentityConflictError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/v1/datasets")
    def datasets():
        return {"datasets": catalog.list_datasets()}

    @app.get("/v1/datasets/{dataset_id}/shards")
    def shards(dataset_id: str):
        return {"shards": [item.__dict__ for item in catalog.list_shards(dataset_id)]}

    @app.get("/v1/datasets/{dataset_id}/episodes")
    def episodes(dataset_id: str):
        return {"episodes": [item.__dict__ for item in catalog.list_episodes(dataset_id)]}

    @app.get("/v1/commits/{commit_id}")
    def receipt(commit_id: str):
        try:
            return catalog.get_receipt(commit_id).__dict__
        except KeyError as error:
            raise HTTPException(404, "commit not found") from error

    @app.post("/v1/datasets/{dataset_id}/snapshots", status_code=201)
    def create_snapshot(dataset_id: str, body: SnapshotRequest):
        try:
            return catalog.create_snapshot(dataset_id, control_source=body.control_source).__dict__
        except KeyError as error:
            raise HTTPException(404, "dataset not found") from error

    @app.get("/v1/snapshots/{snapshot_id}")
    def snapshot(snapshot_id: str):
        try:
            return catalog.get_snapshot(snapshot_id).__dict__
        except KeyError as error:
            raise HTTPException(404, "snapshot not found") from error

    @app.post("/v1/training/jobs", status_code=201)
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

    @app.get("/v1/training/jobs/{job_id}")
    def training_job(job_id: str):
        try:
            return training.get(job_id)
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.get("/v1/training/jobs/{job_id}/logs")
    def training_logs(job_id: str):
        try:
            return {"logs": training.logs(job_id)}
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.get("/v1/training/jobs/{job_id}/metrics")
    def training_metrics(job_id: str):
        try:
            return {"metrics": training.metrics(job_id)}
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.post("/v1/models/revisions", status_code=201)
    def register_model(body: RegisterModelRequest):
        return models.register(**body.model_dump())

    @app.get("/v1/models/revisions/{revision_id}")
    def model_revision(revision_id: str):
        try:
            return models.get(revision_id)
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error

    @app.post("/v1/models/revisions/{revision_id}/validate")
    def validate_model(revision_id: str):
        try:
            return models.validate(revision_id)
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error
        except CandidateError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/deployment")
    def deployment():
        return models.deployment()

    @app.post("/v1/models/revisions/{revision_id}/promote")
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

    @app.post("/v1/deployment/rollback")
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
    return app
