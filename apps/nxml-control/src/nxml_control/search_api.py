from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nxml_control.search import SearchCatalog


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_snapshot_id: str
    embedding_model: str | None = None
    embedding_version: str | None = None


class IndexStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str
    error: str | None = None


class ClipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    episode_id: str
    window_start_ns: int | None = Field(default=None, ge=0)
    window_end_ns: int | None = Field(default=None, ge=0)


class ArtifactStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    error: str | None = None


class ArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip: ClipRequest
    artifact_type: str
    source_snapshot_id: str
    index_version: str
    embedding_model: str | None = None
    embedding_version: str | None = None
    labels: list[str] = []
    derived_features: dict[str, Any] = {}
    status: str = "pending"
    uri: str | None = None


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip: ClipRequest
    labels: list[str]
    source_snapshot_id: str | None = None
    index_version: str | None = None
    status: str = "complete"


class IndexResponse(BaseModel):
    index_version: str
    idempotency_key: str
    source_snapshot_id: str
    embedding_model: str | None
    embedding_version: str | None
    state: str
    error: str | None
    created_at: str
    updated_at: str


class ArtifactResponse(BaseModel):
    artifact_id: str
    idempotency_key: str
    dataset_id: str
    episode_id: str
    window_start_ns: int | None
    window_end_ns: int | None
    artifact_type: str
    embedding_model: str | None
    embedding_version: str | None
    labels: list[str]
    derived_features: dict[str, Any]
    source_snapshot_id: str
    index_version: str
    status: str
    uri: str | None
    error: str | None
    created_at: str


class AnnotationResponse(BaseModel):
    annotation_id: str
    idempotency_key: str
    dataset_id: str
    episode_id: str
    window_start_ns: int
    window_end_ns: int
    labels: list[str]
    source_snapshot_id: str | None
    index_version: str | None
    status: str
    created_at: str


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactResponse]


class AnnotationListResponse(BaseModel):
    annotations: list[AnnotationResponse]


def create_search_router(search: SearchCatalog) -> APIRouter:
    router = APIRouter(prefix="/v1/search", tags=["search metadata"])

    @router.post("/indexes", status_code=201, response_model=IndexResponse)
    def create_index(body: IndexRequest, idempotency_key: str = Header(alias="Idempotency-Key")):
        try:
            return search.create_index(idempotency_key=idempotency_key, **body.model_dump())
        except KeyError as error:
            raise HTTPException(404, "snapshot not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/indexes/{index_version}", response_model=IndexResponse)
    def get_index(index_version: str):
        try:
            return search.get_index(index_version)
        except KeyError as error:
            raise HTTPException(404, "index version not found") from error

    @router.post("/indexes/{index_version}/state", response_model=IndexResponse)
    def set_index_state(index_version: str, body: IndexStateRequest):
        try:
            return search.set_index_state(index_version, **body.model_dump())
        except KeyError as error:
            raise HTTPException(404, "index version not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/artifacts", status_code=201, response_model=ArtifactResponse)
    def create_artifact(
        body: ArtifactRequest, idempotency_key: str = Header(alias="Idempotency-Key")
    ):
        try:
            return search.create_artifact(idempotency_key=idempotency_key, **body.model_dump())
        except KeyError as error:
            raise HTTPException(404, "committed episode, snapshot, or index not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/artifacts/{artifact_id}/state", response_model=ArtifactResponse)
    def set_artifact_state(artifact_id: str, body: ArtifactStateRequest):
        try:
            return search.set_artifact_state(artifact_id, **body.model_dump())
        except KeyError as error:
            raise HTTPException(404, "artifact not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/artifacts/{artifact_id}", response_model=ArtifactResponse)
    def get_artifact(artifact_id: str):
        try:
            return search.get_artifact(artifact_id)
        except KeyError as error:
            raise HTTPException(404, "artifact not found") from error

    @router.get(
        "/episodes/{dataset_id}/{episode_id}/artifacts", response_model=ArtifactListResponse
    )
    def artifacts(dataset_id: str, episode_id: str):
        return {"artifacts": search.artifacts(dataset_id, episode_id)}

    @router.post("/annotations", status_code=201, response_model=AnnotationResponse)
    def create_annotation(
        body: AnnotationRequest, idempotency_key: str = Header(alias="Idempotency-Key")
    ):
        try:
            return search.create_annotation(idempotency_key=idempotency_key, **body.model_dump())
        except KeyError as error:
            raise HTTPException(404, "committed episode not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.get(
        "/episodes/{dataset_id}/{episode_id}/annotations", response_model=AnnotationListResponse
    )
    def annotations(dataset_id: str, episode_id: str):
        return {"annotations": search.annotations(dataset_id, episode_id)}

    return router
