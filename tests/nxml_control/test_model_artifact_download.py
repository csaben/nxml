import base64
import hashlib

from fastapi.testclient import TestClient
from nxml_control.api import create_app

TOKEN = "artifact-token-" + "a" * 48
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def register(client, checkpoint, digest=None):
    checksum = digest or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    response = client.post(
        "/v1/models/revisions",
        headers=AUTH,
        json={
            "model_id": "pokemon-za",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checksum,
            "source_snapshot_id": "sha256:" + "b" * 64,
            "source_config": {},
            "source_commit_id": "source",
            "compatibility": {"action_spec_id": "switch_packets.v1", "action_dim": 26},
            "evaluation": {},
        },
    )
    assert response.status_code == 201
    return response.json()


def test_authenticated_immutable_digest_verified_artifact_download(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoint = root / "policy.pt"
    payload = b"edge-loadable-policy-checkpoint"
    checkpoint.write_bytes(payload)
    client = TestClient(create_app(state_dir=tmp_path, checkpoint_dir=root, auth_token=TOKEN))
    revision = register(client, checkpoint)
    digest = hashlib.sha256(payload).hexdigest()

    assert revision["artifact_id"] == f"sha256:{digest}"
    assert revision["artifact_uri"].endswith(f"/artifacts/sha256:{digest}")
    assert client.get(revision["artifact_uri"]).status_code == 401
    response = client.get(revision["artifact_uri"], headers=AUTH)
    assert response.status_code == 200 and response.content == payload
    assert response.headers["content-length"] == str(len(payload))
    assert response.headers["etag"] == f'"sha256:{digest}"'
    assert response.headers["x-checksum-sha256"] == digest
    assert (
        response.headers["digest"] == "sha-256=" + base64.b64encode(bytes.fromhex(digest)).decode()
    )
    assert "immutable" in response.headers["cache-control"]
    assert client.get(revision["artifact_uri"] + "0", headers=AUTH).status_code == 404
    assert (
        client.get(f"/v1/models/revisions/{revision['revision_id']}", headers=AUTH).json()[
            "artifact_uri"
        ]
        == revision["artifact_uri"]
    )


def test_artifact_missing_corrupt_and_outside_root_fail_closed(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    client = TestClient(create_app(state_dir=tmp_path, checkpoint_dir=root, auth_token=TOKEN))

    missing = root / "missing.pt"
    missing_revision = register(client, missing, "1" * 64)
    assert client.get(missing_revision["artifact_uri"], headers=AUTH).status_code == 404

    corrupt = root / "corrupt.pt"
    corrupt.write_bytes(b"original")
    corrupt_revision = register(client, corrupt)
    corrupt.write_bytes(b"changed-after-registration")
    response = client.get(corrupt_revision["artifact_uri"], headers=AUTH)
    assert response.status_code == 409
    assert "checksum" in response.json()["detail"]

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    outside_revision = register(client, outside)
    response = client.get(outside_revision["artifact_uri"], headers=AUTH)
    assert response.status_code == 403

    symlink = root / "link.pt"
    symlink.symlink_to(outside)
    symlink_revision = register(client, symlink)
    response = client.get(symlink_revision["artifact_uri"], headers=AUTH)
    assert response.status_code == 422
