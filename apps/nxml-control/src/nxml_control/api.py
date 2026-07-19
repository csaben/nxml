from __future__ import annotations

import base64
import hashlib
import hmac
import os
import stat as stat_module
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import FileResponse, JSONResponse

from nxml_control.auth import bearer_matches
from nxml_control.catalog import Catalog, IdentityConflictError, InvalidManifestError, Upload
from nxml_control.models import (
    CandidateError,
    ConflictError,
    DeploymentRuntime,
    FakePolicyRuntime,
    ModelRegistry,
)
from nxml_control.search import SearchCatalog
from nxml_control.search_api import create_search_router
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage
from nxml_control.training import FakeTrainingExecutor, TrainingExecutor, TrainingJobs, TrainingSpec


class HealthResponse(BaseModel):
    status: str


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
    artifact_id: str
    artifact_uri: str
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_snapshot_id: str
    source_config: dict[str, Any]
    source_commit_id: str
    compatibility: dict[str, Any]
    evaluation: dict[str, Any]
    validation: dict[str, Any] | None
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


class SnapshotListResponse(BaseModel):
    snapshots: list[SnapshotResponse]


class TrainingJobListResponse(BaseModel):
    jobs: list[TrainingJobResponse]


class ModelRevisionListResponse(BaseModel):
    revisions: list[ModelRevisionResponse]


class DeploymentListResponse(BaseModel):
    deployments: list[DeploymentResponse]


class CommitReceiptListResponse(BaseModel):
    receipts: list[CommitReceiptResponse]


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


class TrainingArtifactResponse(BaseModel):
    artifact_type: str
    uri: str
    sha256: str
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingArtifactsResponse(BaseModel):
    artifacts: list[TrainingArtifactResponse]


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
    training_job_id: str | None = None


class ActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_generation: int | None = Field(default=None, ge=0)
    expected_revision: str | None = None


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config: dict[str, Any]


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control_source: str = Field(default="all", pattern=r"^(all|human|policy)$")


class EpisodeQualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_id: str = Field(
        default="nxml.episode-quality.v1", pattern=r"^nxml\.episode-quality\.v1$"
    )
    training_eligible: bool
    reason: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    validator_version: str = Field(min_length=1)


class EpisodeQualityResponse(BaseModel):
    disposition_id: str
    idempotency_key: str
    dataset_id: str
    episode_id: str
    schema_id: str
    training_eligible: bool
    reason: str
    validator: str
    validator_version: str
    created_at: str


class EpisodeQualityListResponse(BaseModel):
    dispositions: list[EpisodeQualityResponse]


class CommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(min_length=1)
    shard_id: str = Field(min_length=1)
    manifest: dict[str, Any]


def _view(item: Upload) -> dict[str, Any]:
    return {**item.__dict__, "upload_url": f"/v1/uploads/{item.id}/content"}


def _model_view(item: dict[str, Any]) -> dict[str, Any]:
    artifact_id = "sha256:" + item["checkpoint_sha256"]
    return {
        **item,
        "artifact_id": artifact_id,
        "artifact_uri": f"/v1/models/revisions/{item['revision_id']}/artifacts/{artifact_id}",
    }


def create_app(
    *,
    state_dir: str | Path,
    training_executor: TrainingExecutor | None = None,
    deployment_runtime: DeploymentRuntime | None = None,
    auth_token: str | None = None,
    training_async: bool = False,
    checkpoint_dir: str | Path | None = None,
    allow_fake_deployment_runtime: bool = True,
) -> FastAPI:
    state = Path(state_dir)
    artifact_root = Path(checkpoint_dir or state / "checkpoints").resolve()
    catalog = Catalog(state / "catalog.sqlite3")
    service = IngestService(catalog, LocalObjectStorage(state / "objects"))
    training = TrainingJobs(
        state / "catalog.sqlite3",
        training_executor or FakeTrainingExecutor(),
        run_async=training_async,
    )
    if deployment_runtime is None:
        if not allow_fake_deployment_runtime:
            raise RuntimeError("production requires a real deployment validation runtime")
        deployment_runtime = FakePolicyRuntime()
    if not allow_fake_deployment_runtime and isinstance(deployment_runtime, FakePolicyRuntime):
        raise RuntimeError("FakePolicyRuntime is forbidden in production")
    models = ModelRegistry(state / "catalog.sqlite3", deployment_runtime)
    search = SearchCatalog(state / "catalog.sqlite3")
    app = FastAPI(title="NXML ML Control Plane", version="1.0.0")

    if auth_token is not None:
        if len(auth_token) < 32:
            raise ValueError("auth_token must be at least 32 characters")

        @app.middleware("http")
        async def require_bearer(request: Request, call_next):
            if request.url.path != "/healthz" and not bearer_matches(
                request.headers.get("Authorization"), auth_token
            ):
                return JSONResponse(
                    {"detail": "invalid or missing bearer token"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    app.include_router(create_search_router(search))

    @app.get("/healthz", response_model=HealthResponse)
    def health():
        return {"status": "ready"}

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

    @app.post(
        "/v1/datasets/{dataset_id}/episodes/{episode_id}/quality-dispositions",
        status_code=201,
        response_model=EpisodeQualityResponse,
    )
    def set_episode_quality(
        dataset_id: str,
        episode_id: str,
        body: EpisodeQualityRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    ):
        try:
            fields = body.model_dump()
            fields.pop("schema_id")
            return catalog.set_episode_quality(
                dataset_id, episode_id, idempotency_key=idempotency_key, **fields
            )
        except KeyError as error:
            raise HTTPException(404, "committed episode not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get(
        "/v1/datasets/{dataset_id}/episodes/{episode_id}/quality-dispositions",
        response_model=EpisodeQualityListResponse,
    )
    def episode_quality(dataset_id: str, episode_id: str):
        return {"dispositions": catalog.episode_quality(dataset_id, episode_id)}

    @app.get("/v1/commits", response_model=CommitReceiptListResponse)
    def receipts(
        dataset_id: str | None = None,
        shard_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        limit = min(max(limit, 1), 500)
        return {
            "receipts": [
                item.__dict__
                for item in catalog.list_receipts(
                    dataset_id=dataset_id,
                    shard_id=shard_id,
                    limit=limit,
                    offset=max(offset, 0),
                )
            ]
        }

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
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/snapshots", response_model=SnapshotListResponse)
    def snapshots(
        dataset_id: str | None = None,
        control_source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        limit = min(max(limit, 1), 500)
        return {
            "snapshots": [
                item.__dict__
                for item in catalog.list_snapshots(
                    dataset_id=dataset_id,
                    control_source=control_source,
                    limit=limit,
                    offset=max(offset, 0),
                )
            ]
        }

    @app.get("/v1/snapshots/{snapshot_id}", response_model=SnapshotResponse)
    def snapshot(snapshot_id: str):
        try:
            return catalog.get_snapshot(snapshot_id).__dict__
        except KeyError as error:
            raise HTTPException(404, "snapshot not found") from error

    @app.get("/v1/training/jobs", response_model=TrainingJobListResponse)
    def training_jobs(
        state: str | None = None, snapshot_id: str | None = None, limit: int = 100, offset: int = 0
    ):
        limit = min(max(limit, 1), 500)
        return {
            "jobs": training.list(
                state=state, snapshot_id=snapshot_id, limit=limit, offset=max(offset, 0)
            )
        }

    @app.post("/v1/training/jobs", status_code=201, response_model=TrainingJobResponse)
    def submit_training(
        body: TrainingRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    ):
        if not training.available:
            raise HTTPException(503, "training executor is not configured")
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

    @app.get("/v1/training/jobs/{job_id}/artifacts", response_model=TrainingArtifactsResponse)
    def training_artifacts(job_id: str):
        try:
            return {"artifacts": training.artifacts(job_id)}
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error

    @app.post("/v1/training/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
    def cancel_training(job_id: str):
        try:
            return training.cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, "training job not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/v1/models/revisions", response_model=ModelRevisionListResponse)
    def model_revisions(
        model_id: str | None = None, state: str | None = None, limit: int = 100, offset: int = 0
    ):
        limit = min(max(limit, 1), 500)
        return {
            "revisions": [
                _model_view(item)
                for item in models.list_revisions(
                    model_id=model_id, state=state, limit=limit, offset=max(offset, 0)
                )
            ]
        }

    @app.post("/v1/models/revisions", status_code=201, response_model=ModelRevisionResponse)
    def register_model(body: RegisterModelRequest):
        fields = body.model_dump()
        training_job_id = fields.pop("training_job_id")
        if training_job_id is not None:
            try:
                job = training.get(training_job_id)
            except KeyError as error:
                raise HTTPException(404, "training job not found") from error
            if job["state"] != "succeeded":
                raise HTTPException(409, "training job has not succeeded")
            if (
                job["snapshot_id"] != body.source_snapshot_id
                or job["checkpoint_path"] != body.checkpoint_path
                or job["checkpoint_sha256"] != body.checkpoint_sha256
            ):
                raise HTTPException(422, "model revision does not match verified training artifact")
        return _model_view(models.register(**fields))

    @app.get("/v1/models/revisions/{revision_id}", response_model=ModelRevisionResponse)
    def model_revision(revision_id: str):
        try:
            return _model_view(models.get(revision_id))
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error

    @app.get(
        "/v1/models/revisions/{revision_id}/artifacts/{artifact_id}",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
                },
                "description": "Digest-verified immutable policy checkpoint",
            }
        },
    )
    def download_model_artifact(revision_id: str, artifact_id: str):
        try:
            revision = models.get(revision_id)
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error
        expected_id = "sha256:" + revision["checkpoint_sha256"]
        if artifact_id != expected_id:
            raise HTTPException(404, "model artifact not found")
        raw_path = Path(revision["checkpoint_path"])
        if not raw_path.is_absolute():
            raise HTTPException(422, "model artifact path is not an absolute managed path")
        try:
            resolved = raw_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise HTTPException(404, "model artifact file not found") from error
        if raw_path != resolved:
            raise HTTPException(422, "model artifact path must be canonical and symlink-free")
        try:
            resolved.relative_to(artifact_root)
        except ValueError as error:
            raise HTTPException(
                403, "model artifact is outside the managed checkpoint root"
            ) from error
        file_stat = resolved.stat()
        if not stat_module.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.geteuid():
            raise HTTPException(403, "model artifact ownership or type is invalid")
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        if not hmac.compare_digest(actual.encode(), revision["checkpoint_sha256"].encode()):
            raise HTTPException(409, "model artifact checksum does not match immutable revision")
        digest_header = base64.b64encode(bytes.fromhex(actual)).decode()
        return FileResponse(
            resolved,
            media_type="application/octet-stream",
            filename=f"{revision_id}.pt",
            headers={
                "ETag": f'"sha256:{actual}"',
                "Digest": f"sha-256={digest_header}",
                "X-Checksum-SHA256": actual,
                "Cache-Control": "private, max-age=31536000, immutable",
                "Content-Length": str(file_stat.st_size),
            },
        )

    @app.post("/v1/models/revisions/{revision_id}/validate", response_model=ModelRevisionResponse)
    def validate_model(revision_id: str):
        try:
            return _model_view(models.validate(revision_id))
        except KeyError as error:
            raise HTTPException(404, "model revision not found") from error
        except CandidateError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/deployments", response_model=DeploymentListResponse)
    def deployments():
        return {"deployments": [models.deployment()]}

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
                expected_generation=body.expected_generation,
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
                expected_revision=body.expected_revision,
                expected_generation=body.expected_generation,
                idempotency_key=idempotency_key,
            )
        except CandidateError as error:
            raise HTTPException(422, str(error)) from error
        except ConflictError as error:
            raise HTTPException(409, str(error)) from error

    app.state.catalog = catalog
    app.state.ingest = service
    app.state.training = training
    app.state.models = models
    app.state.search = search
    app.state.ingest = service

    def secured_openapi():
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
        }
        for path, item in schema["paths"].items():
            if path == "/healthz":
                continue
            for operation in item.values():
                if isinstance(operation, dict) and "responses" in operation:
                    operation["security"] = [{"bearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = secured_openapi
    return app
