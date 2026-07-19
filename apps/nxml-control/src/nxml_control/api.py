from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from nxml_control.catalog import Catalog, Upload
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage


class CreateUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_key: str = Field(pattern=r"^uploads/[A-Za-z0-9._/-]+\.tar$")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(min_length=1)
    shard_id: str = Field(min_length=1)
    manifest: dict[str, Any]


def _view(item: Upload) -> dict[str, Any]:
    return {**item.__dict__, "upload_url": f"/v1/uploads/{item.id}/content"}


def create_app(*, state_dir: str | Path) -> FastAPI:
    state = Path(state_dir)
    catalog = Catalog(state / "catalog.sqlite3")
    service = IngestService(catalog, LocalObjectStorage(state / "objects"))
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
            return _view(catalog.commit(upload_id, **body.model_dump()))
        except KeyError as error:
            raise HTTPException(404, "upload not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    app.state.catalog = catalog
    app.state.ingest = service
    return app
