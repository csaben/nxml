import hashlib
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.catalog import Catalog
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage


def components(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    return catalog, IngestService(catalog, LocalObjectStorage(tmp_path / "objects"))


def test_checksum_gated_exactly_once_commit(tmp_path):
    catalog, service = components(tmp_path)
    content = b"immutable webdataset"
    digest = hashlib.sha256(content).hexdigest()
    request = dict(
        idempotency_key="capture/shard",
        object_key="uploads/capture/shard.tar",
        size_bytes=len(content),
        sha256=digest,
    )
    created = catalog.create_upload(**request)
    assert catalog.create_upload(**request).id == created.id
    with pytest.raises(ValueError, match="verified"):
        catalog.commit(created.id, dataset_id="raw", shard_id="s1", manifest={})
    assert service.upload(created.id, BytesIO(content)).state == "uploaded"
    committed = catalog.commit(
        created.id, dataset_id="raw", shard_id="s1", manifest={"episodes": ["e1"]}
    )
    assert committed.state == "committed"
    assert (
        catalog.commit(created.id, dataset_id="raw", shard_id="s1", manifest={}).id == committed.id
    )


def test_changed_idempotent_request_and_bad_checksum_fail(tmp_path):
    catalog, service = components(tmp_path)
    upload = catalog.create_upload(
        idempotency_key="same", object_key="uploads/a.tar", size_bytes=3, sha256="0" * 64
    )
    with pytest.raises(ValueError, match="different request"):
        catalog.create_upload(
            idempotency_key="same", object_key="uploads/a.tar", size_bytes=4, sha256="1" * 64
        )
    with pytest.raises(ValueError, match="checksum"):
        service.upload(upload.id, BytesIO(b"bad"))
    assert catalog.get(upload.id).state == "created"


def test_http_contract_and_openapi(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content = b"tar"
    digest = hashlib.sha256(content).hexdigest()
    response = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "edge/shard"},
        json={"object_key": "uploads/edge/shard.tar", "size_bytes": 3, "sha256": digest},
    )
    assert response.status_code == 201
    upload = response.json()
    assert client.put(upload["upload_url"], content=content).json()["state"] == "uploaded"
    assert (
        client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "raw", "shard_id": "edge-1", "manifest": {}},
        ).json()["state"]
        == "committed"
    )
    assert client.get("/healthz").json()["committed"] == 1
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/uploads/{upload_id}/inspect" in paths


def test_existing_object_key_is_immutable(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    first = storage.put_if_absent("uploads/a.tar", BytesIO(b"first"))
    second = storage.put_if_absent("uploads/a.tar", BytesIO(b"second"))
    assert second == first
    with storage.open("uploads/a.tar") as source:
        assert source.read() == b"first"
