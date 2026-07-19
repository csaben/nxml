import hashlib
import io
import tarfile

import pytest
from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.catalog import Catalog
from nxml_control.segments import (
    SEGMENT_ROUTE_CONTRACTS,
    SegmentCatalog,
    SegmentConflictError,
    SegmentContractError,
    SegmentNotFoundError,
)
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage

RUN_ID = "11111111-1111-4111-8111-111111111111"
BAD_ID = "22222222-2222-4222-8222-222222222222"
QUALITY_ID = "33333333-3333-4333-8333-333333333333"
UNSAFE_ID = "44444444-4444-4444-8444-444444444444"


def segment_tar(episode, index, start, end):
    members = {
        f"{episode}.{index:06d}.mkv": f"video-{index}".encode(),
        f"{episode}.{index:06d}.parquet": f"actions-{index}".encode(),
        f"{episode}.{index:06d}.events.parquet": f"events-{index}".encode(),
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for path, content in members.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    content = stream.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_id": "nxml.segment-bundle.v1",
        "dataset_id": "pokemon",
        "episode_id": episode,
        "segment_id": "sha256:" + digest,
        "sequence_index": index,
        "clock_id": "linux-monotonic",
        "timeline_start_ns": start,
        "timeline_end_ns": end,
        "object_size_bytes": len(content),
        "object_sha256": digest,
        "members": [
            {
                "role": "events"
                if path.endswith(".events.parquet")
                else ("actions" if path.endswith(".parquet") else "video"),
                "path": path,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for path, data in members.items()
        ],
    }
    return content, manifest


def system(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    storage = LocalObjectStorage(tmp_path / "objects")
    ingest = IngestService(catalog, storage)
    return catalog, ingest, SegmentCatalog(catalog, ingest, storage)


def upload(catalog, ingest, segments, content, manifest, key):
    item = catalog.create_upload(
        idempotency_key=key,
        object_key=(
            f"uploads/segments/{manifest['episode_id']}/"
            f"{manifest['sequence_index']:06d}-{manifest['object_sha256']}.tar"
        ),
        size_bytes=len(content),
        sha256=manifest["object_sha256"],
    )
    ingest.upload(item.id, io.BytesIO(content))
    return segments.commit_segment(item.id, manifest)


def close_body(manifests):
    ordered = sorted(manifests, key=lambda item: item["sequence_index"])
    return {
        "schema_id": "nxml.episode-close.v1",
        "dataset_id": "pokemon",
        "episode_id": ordered[0]["episode_id"],
        "clock_id": "linux-monotonic",
        "timeline_start_ns": ordered[0]["timeline_start_ns"],
        "timeline_end_ns": ordered[-1]["timeline_end_ns"],
        "segments": [
            {
                "segment_id": item["segment_id"],
                "sequence_index": item["sequence_index"],
                "timeline_start_ns": item["timeline_start_ns"],
                "timeline_end_ns": item["timeline_end_ns"],
                "object_sha256": item["object_sha256"],
            }
            for item in ordered
        ],
    }


def test_segment_crash_retry_receipt_and_reconstruction_are_idempotent(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content0, manifest0 = segment_tar(RUN_ID, 0, 100, 200)
    pending = catalog.create_upload(
        idempotency_key="segment-0",
        object_key=(f"uploads/segments/{RUN_ID}/000000-{manifest0['object_sha256']}.tar"),
        size_bytes=len(content0),
        sha256=manifest0["object_sha256"],
    )
    ingest.upload(pending.id, io.BytesIO(content0))
    assert segments.status("pokemon")["committed_segments"] == 0

    # Process restart after durable bytes but before registration.
    segments = SegmentCatalog(catalog, ingest, ingest.storage)
    receipt0 = segments.commit_segment(pending.id, manifest0)
    assert segments.commit_segment(pending.id, manifest0) == receipt0

    content1, manifest1 = segment_tar(RUN_ID, 1, 200, 300)
    receipt1 = upload(catalog, ingest, segments, content1, manifest1, "segment-1")
    assert receipt1.sequence_index == 1
    close = segments.close_episode(close_body([manifest1, manifest0]), idempotency_key="close-run")
    assert (
        segments.close_episode(close_body([manifest0, manifest1]), idempotency_key="close-run")[
            "close_id"
        ]
        == close["close_id"]
    )
    reconstructed = list(segments.iter_reconstruction(close["close_id"]))
    assert [item[0]["sequence_index"] for item in reconstructed] == [0, 1]
    assert b"".join(item[1] for item in reconstructed if item[0]["sequence_index"] == 0) == content0
    assert b"".join(item[1] for item in reconstructed if item[0]["sequence_index"] == 1) == content1
    status = segments.status("pokemon")
    assert status["committed_segments"] == 2
    assert status["committed_bytes"] == len(content0) + len(content1)
    assert status["closed_episodes"] == 1
    assert status["eligible_episodes"] == 0
    assert status["excluded_episodes"] == 1


def test_episode_close_rejects_gap_overlap_missing_and_digest_mismatch(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    _, first = segment_tar(BAD_ID, 0, 0, 10)
    _, second = segment_tar(BAD_ID, 1, 10, 20)
    for index, manifest in enumerate((first, second)):
        content, _ = segment_tar(BAD_ID, index, index * 10, (index + 1) * 10)
        upload(catalog, ingest, segments, content, manifest, f"bad-{index}")

    missing = close_body([first, second])
    missing["segments"][1]["segment_id"] = "sha256:" + "0" * 64
    with pytest.raises(SegmentNotFoundError, match="missing segment") as missing_error:
        segments.close_episode(missing, idempotency_key="missing")
    assert missing_error.value.status_code == 404

    digest = close_body([first, second])
    digest["segments"][1]["object_sha256"] = "1" * 64
    with pytest.raises(SegmentContractError, match="metadata mismatch"):
        segments.close_episode(digest, idempotency_key="digest")

    gap = close_body([first, second])
    gap["segments"][1]["timeline_start_ns"] = 11
    with pytest.raises(SegmentContractError, match="gap"):
        segments.close_episode(gap, idempotency_key="gap")

    overlap = close_body([first, second])
    overlap["segments"][1]["timeline_start_ns"] = 9
    with pytest.raises(SegmentContractError, match="overlap"):
        segments.close_episode(overlap, idempotency_key="overlap")

    order = close_body([first, second])
    order["segments"][1]["sequence_index"] = 2
    with pytest.raises(SegmentContractError, match="contiguous"):
        segments.close_episode(order, idempotency_key="order")


def test_segment_quality_propagates_to_snapshot_without_mutating_receipt(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content, manifest = segment_tar(QUALITY_ID, 0, 0, 100)
    receipt = upload(catalog, ingest, segments, content, manifest, "quality-segment")
    close = segments.close_episode(close_body([manifest]), idempotency_key="quality-close")
    disposition = segments.set_quality(
        manifest["segment_id"],
        idempotency_key="quality-v1",
        training_eligible=False,
        reason="noncausal_action_alignment",
        validator="alignment",
        validator_version="1",
        validator_state="failed",
    )
    assert (
        segments.set_quality(
            manifest["segment_id"],
            idempotency_key="quality-v1",
            training_eligible=False,
            reason="noncausal_action_alignment",
            validator="alignment",
            validator_version="1",
            validator_state="failed",
        )["disposition_id"]
        == disposition["disposition_id"]
    )
    snapshot = segments.create_snapshot("pokemon")
    assert snapshot["episodes"] == []
    exclusion = snapshot["excluded_episodes"][0]
    assert exclusion["episode_id"] == QUALITY_ID
    assert exclusion["segments"][0]["reason"] == "noncausal_action_alignment"
    assert segments.get_receipt(receipt.receipt_id) == receipt
    assert (
        b"".join(
            chunk for _, chunk in segments.iter_reconstruction(close["close_id"], chunk_size=7)
        )
        == content
    )

    segments.set_quality(
        manifest["segment_id"],
        idempotency_key="quality-v2",
        training_eligible=True,
        reason="alignment_corrected",
        validator="alignment",
        validator_version="2",
        validator_state="passed",
    )
    assert segments.get_snapshot(snapshot["snapshot_id"]) == snapshot
    replacement = segments.create_snapshot("pokemon")
    assert replacement["snapshot_id"] != snapshot["snapshot_id"]
    assert replacement["episodes"][0]["episode_id"] == QUALITY_ID


def test_segment_member_rejects_unsafe_path(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content, manifest = segment_tar(UNSAFE_ID, 0, 0, 10)
    manifest["members"][0]["path"] = "../events.parquet"
    pending = catalog.create_upload(
        idempotency_key="unsafe",
        object_key=(f"uploads/segments/{UNSAFE_ID}/000000-{manifest['object_sha256']}.tar"),
        size_bytes=len(content),
        sha256=manifest["object_sha256"],
    )
    ingest.upload(pending.id, io.BytesIO(content))
    with pytest.raises(SegmentContractError, match="safe relative path"):
        segments.commit_segment(pending.id, manifest)

    _, manifest = segment_tar(UNSAFE_ID, 0, 0, 10)
    manifest["members"][0]["path"] = "different.000000.events.parquet"
    with pytest.raises(SegmentContractError, match="reuse episode UUID"):
        segments.commit_segment("unused", manifest)


def test_snapshot_fails_closed_without_explicit_passing_validator(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content, manifest = segment_tar(RUN_ID, 0, 0, 10)
    upload(catalog, ingest, segments, content, manifest, "unvalidated")
    segments.close_episode(close_body([manifest]), idempotency_key="close-unvalidated")

    snapshot = segments.create_snapshot("pokemon")
    assert snapshot["episodes"] == []
    exclusion = snapshot["excluded_episodes"][0]["segments"][0]
    assert exclusion["reason"] == "missing_quality_disposition"
    assert exclusion["validator_state"] == "missing"

    segments.set_quality(
        manifest["segment_id"],
        idempotency_key="validated",
        training_eligible=True,
        reason="validated",
        validator="segment-validator",
        validator_version="1",
        validator_state="passed",
    )
    assert segments.create_snapshot("pokemon")["episodes"][0]["episode_id"] == RUN_ID


def test_typed_errors_staged_mapping_and_exportable_manifest(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content, manifest = segment_tar(RUN_ID, 0, 0, 10)
    bad_key = catalog.create_upload(
        idempotency_key="legacy-key",
        object_key="uploads/legacy.tar",
        size_bytes=len(content),
        sha256=manifest["object_sha256"],
    )
    ingest.upload(bad_key.id, io.BytesIO(content))
    with pytest.raises(SegmentContractError) as invalid:
        segments.commit_segment(bad_key.id, manifest)
    assert invalid.value.status_code == 422

    receipt = upload(catalog, ingest, segments, content, manifest, "valid-key")
    exported = segments.get_segment(receipt.segment_id)
    assert exported["manifest"] == manifest
    assert len(exported["manifest"]["members"]) == 3
    assert exported["receipt"]["storage_key"].startswith("uploads/segments/")

    with pytest.raises(SegmentNotFoundError) as missing:
        segments.get_receipt("missing")
    assert missing.value.status_code == 404
    with pytest.raises(SegmentNotFoundError):
        segments.quality("sha256:" + "0" * 64)

    with pytest.raises(SegmentConflictError) as conflict:
        segments.set_quality(
            receipt.segment_id,
            idempotency_key="quality-conflict",
            training_eligible=True,
            reason="validated",
            validator="segment-validator",
            validator_version="1",
            validator_state="passed",
        )
        segments.set_quality(
            receipt.segment_id,
            idempotency_key="quality-conflict",
            training_eligible=False,
            reason="changed",
            validator="segment-validator",
            validator_version="1",
            validator_state="failed",
        )
    assert conflict.value.status_code == 409


def test_service_boundary_declares_auditable_segment_routes():
    assert ("GET", "/v1/segments/{segment_id}/quality-dispositions") in SEGMENT_ROUTE_CONTRACTS
    assert ("POST", "/v1/datasets/{dataset_id}/segment-snapshots") in SEGMENT_ROUTE_CONTRACTS
    assert ("GET", "/v1/segment-snapshots/{snapshot_id}") in SEGMENT_ROUTE_CONTRACTS


def test_segment_routes_survive_restart_and_receipt_gates_source_delete(tmp_path):
    content0, manifest0 = segment_tar(RUN_ID, 0, 100, 200)
    content1, manifest1 = segment_tar(RUN_ID, 1, 200, 300)
    client = TestClient(create_app(state_dir=tmp_path))

    receipts = []
    for index, (content, manifest) in enumerate(((content0, manifest0), (content1, manifest1))):
        prepared = client.post(
            "/v1/uploads",
            headers={"Idempotency-Key": f"segment-route-{index}"},
            json={
                "object_key": (
                    f"uploads/segments/{RUN_ID}/{index:06d}-{manifest['object_sha256']}.tar"
                ),
                "size_bytes": len(content),
                "sha256": manifest["object_sha256"],
            },
        )
        assert prepared.status_code == 201
        upload_id = prepared.json()["id"]
        assert client.put(f"/v1/uploads/{upload_id}/content", content=content).status_code == 200
        assert client.post(f"/v1/uploads/{upload_id}/inspect").json()["state"] == "uploaded"

        # Simulate a controller crash after durable bytes and inspection, before commit.
        client = TestClient(create_app(state_dir=tmp_path))
        committed = client.post(f"/v1/segment-bundles/{upload_id}/commit", json=manifest)
        assert committed.status_code == 200
        receipt = committed.json()
        assert receipt["state"] == "committed"
        assert receipt["sha256"] == manifest["object_sha256"]
        assert (
            client.post(f"/v1/segment-bundles/{upload_id}/commit", json=manifest).json() == receipt
        )
        assert client.get(f"/v1/segment-receipts/{receipt['receipt_id']}").json() == receipt
        receipts.append(receipt)

    # The producer may discard both staged/source bytes only after the receipt above.
    del content0, content1
    close_request = close_body([manifest1, manifest0])
    close = client.post(
        f"/v1/datasets/pokemon/episodes/{RUN_ID}/close",
        headers={"Idempotency-Key": "segment-route-close"},
        json=close_request,
    )
    assert close.status_code == 201
    close_body_response = close.json()
    assert close_body_response["manifest"]["segments"][0]["sequence_index"] == 0
    assert (
        client.post(
            f"/v1/datasets/pokemon/episodes/{RUN_ID}/close",
            headers={"Idempotency-Key": "segment-route-close"},
            json=close_request,
        ).json()
        == close_body_response
    )

    excluded = client.post("/v1/datasets/pokemon/segment-snapshots")
    assert excluded.status_code == 201
    assert excluded.json()["episodes"] == []
    assert excluded.json()["excluded_episodes"][0]["segments"][0]["reason"] == (
        "missing_quality_disposition"
    )

    for receipt in receipts:
        quality = client.post(
            f"/v1/segments/{receipt['segment_id']}/quality-dispositions",
            headers={"Idempotency-Key": f"quality-{receipt['sequence_index']}"},
            json={
                "schema_id": "nxml.segment-quality.v1",
                "training_eligible": True,
                "reason": "strict_action_and_media_validation_passed",
                "validator": "nxml-segment-validator",
                "validator_version": "1",
                "validator_state": "passed",
            },
        )
        assert quality.status_code == 201

    episode_quality_url = f"/v1/datasets/pokemon/episodes/{RUN_ID}/quality-dispositions"
    episode_veto = client.post(
        episode_quality_url,
        headers={"Idempotency-Key": "rolling-episode-veto"},
        json={
            "schema_id": "nxml.episode-quality.v1",
            "training_eligible": False,
            "reason": "synthetic_canary",
            "validator": "edge-canary",
            "validator_version": "1",
        },
    )
    assert episode_veto.status_code == 201
    assert (
        client.post(
            episode_quality_url,
            headers={"Idempotency-Key": "rolling-episode-veto"},
            json={
                "schema_id": "nxml.episode-quality.v1",
                "training_eligible": False,
                "reason": "synthetic_canary",
                "validator": "edge-canary",
                "validator_version": "1",
            },
        ).json()
        == episode_veto.json()
    )
    assert client.get(episode_quality_url).json()["dispositions"] == [episode_veto.json()]
    vetoed = client.post("/v1/datasets/pokemon/segment-snapshots").json()
    assert vetoed["episodes"] == []
    assert vetoed["excluded_episodes"][0]["episode_disposition"]["reason"] == ("synthetic_canary")
    assert vetoed["excluded_episodes"][0]["segments"] == []

    episode_pass = client.post(
        episode_quality_url,
        headers={"Idempotency-Key": "rolling-episode-pass"},
        json={
            "schema_id": "nxml.episode-quality.v1",
            "training_eligible": True,
            "reason": "real_episode_validated",
            "validator": "edge-canary",
            "validator_version": "2",
        },
    )
    assert episode_pass.status_code == 201

    included = client.post("/v1/datasets/pokemon/segment-snapshots")
    assert included.status_code == 201
    assert included.json()["episodes"][0]["segment_ids"] == [
        manifest0["segment_id"],
        manifest1["segment_id"],
    ]
    assert (
        client.get(f"/v1/segment-snapshots/{included.json()['snapshot_id']}").json()
        == included.json()
    )

    reconstructed = list(
        client.app.state.segments.iter_reconstruction(close_body_response["close_id"], chunk_size=7)
    )
    assert reconstructed
    assert max(len(chunk) for _, chunk in reconstructed) <= 7
    assert [manifest["sequence_index"] for manifest, _ in reconstructed[:1]] == [0]
    status = client.get("/v1/datasets/pokemon/segment-status").json()
    assert {
        key: status[key]
        for key in (
            "schema_id",
            "dataset_id",
            "committed_segments",
            "committed_bytes",
            "closed_episodes",
            "eligible_episodes",
            "excluded_episodes",
            "latest_segment_committed_at",
        )
    } == {
        "schema_id": "nxml.segment-status.v1",
        "dataset_id": "pokemon",
        "committed_segments": 2,
        "committed_bytes": manifest0["object_size_bytes"] + manifest1["object_size_bytes"],
        "closed_episodes": 1,
        "eligible_episodes": 1,
        "excluded_episodes": 0,
        "latest_segment_committed_at": receipts[-1]["committed_at"],
    }
    assert status["durable_object_bytes"] == status["committed_bytes"]
    assert status["in_progress_upload_bytes"] == 0
    assert status["episode_count"] == 1
    assert status["ingest_admission"] == {
        "state": "admitting",
        "reason": "capacity_available",
    }
    assert {item["role"] for item in status["filesystems"]} == {
        "authoritative_objects",
        "upload_staging",
        "temporary_spool",
        "catalog",
    }
    assert all("/" not in item["filesystem_label"] for item in status["filesystems"])

    paths = client.get("/openapi.json").json()["paths"]
    for _, path in SEGMENT_ROUTE_CONTRACTS:
        assert path in paths


def test_segment_routes_map_missing_invalid_and_conflict_statuses(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, manifest = segment_tar(RUN_ID, 0, 0, 10)
    missing = client.post("/v1/segment-bundles/missing/commit", json=manifest)
    assert missing.status_code == 404

    request = {
        "object_key": f"uploads/segments/{RUN_ID}/000000-{manifest['object_sha256']}.tar",
        "size_bytes": len(content),
        "sha256": manifest["object_sha256"],
    }
    upload_id = client.post(
        "/v1/uploads", headers={"Idempotency-Key": "typed-segment"}, json=request
    ).json()["id"]
    client.put(f"/v1/uploads/{upload_id}/content", content=content)
    client.post(f"/v1/uploads/{upload_id}/inspect")
    bad = dict(manifest)
    bad["timeline_end_ns"] = bad["timeline_start_ns"]
    assert client.post(f"/v1/segment-bundles/{upload_id}/commit", json=bad).status_code == 422
    assert client.post(f"/v1/segment-bundles/{upload_id}/commit", json=manifest).status_code == 200

    conflict = client.post(
        f"/v1/segments/{manifest['segment_id']}/quality-dispositions",
        headers={"Idempotency-Key": "segment-quality-conflict"},
        json={
            "training_eligible": True,
            "reason": "passed",
            "validator": "validator",
            "validator_version": "1",
            "validator_state": "passed",
        },
    )
    assert conflict.status_code == 201
    changed = client.post(
        f"/v1/segments/{manifest['segment_id']}/quality-dispositions",
        headers={"Idempotency-Key": "segment-quality-conflict"},
        json={
            "training_eligible": False,
            "reason": "changed",
            "validator": "validator",
            "validator_version": "1",
            "validator_state": "failed",
        },
    )
    assert changed.status_code == 409


def test_storage_capacity_blocks_new_ingest_but_preserves_reads(tmp_path, monkeypatch):
    client = TestClient(create_app(state_dir=tmp_path, ingest_reserved_bytes=10**30))
    status = client.get("/v1/datasets/pokemon/segment-status")
    assert status.status_code == 200
    body = status.json()
    assert body["ingest_admission"] == {
        "state": "blocked",
        "reason": "insufficient_available_bytes",
    }
    assert body["reserved_headroom_bytes"] == 10**30
    assert client.get("/v1/datasets").status_code == 200
    blocked = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "capacity-blocked"},
        json={"object_key": "uploads/blocked.tar", "size_bytes": 1, "sha256": "0" * 64},
    )
    assert blocked.status_code == 507
    assert blocked.json()["detail"]["state"] == "blocked"

    import nxml_control.api as api_module

    def stat_failed(_path):
        raise OSError("simulated stat failure")

    monkeypatch.setattr(api_module.os, "statvfs", stat_failed)
    failed = client.get("/v1/datasets/pokemon/segment-status")
    assert failed.status_code == 200
    assert failed.json()["ingest_admission"]["state"] == "blocked"
    assert "stat_failed" in failed.json()["ingest_admission"]["reason"]
    assert client.get("/v1/datasets").status_code == 200
    blocked_stat = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "capacity-stat-failed"},
        json={"object_key": "uploads/stat-failed.tar", "size_bytes": 1, "sha256": "1" * 64},
    )
    assert blocked_stat.status_code == 507


def test_upload_body_is_size_bounded_before_object_registration(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "bounded-body"},
        json={"object_key": "uploads/bounded.tar", "size_bytes": 4, "sha256": "0" * 64},
    ).json()
    assert client.put(f"/v1/uploads/{upload['id']}/content", content=b"12345").status_code == 422
    assert client.post(f"/v1/uploads/{upload['id']}/inspect").json()["state"] == "created"


def test_existing_upload_retry_survives_capacity_block(tmp_path):
    request = {"object_key": "uploads/retry.tar", "size_bytes": 1, "sha256": "2" * 64}
    headers = {"Idempotency-Key": "capacity-response-loss"}
    admitting = TestClient(create_app(state_dir=tmp_path))
    original = admitting.post("/v1/uploads", headers=headers, json=request)
    assert original.status_code == 201

    blocked = TestClient(create_app(state_dir=tmp_path, ingest_reserved_bytes=10**30))
    retry = blocked.post("/v1/uploads", headers=headers, json=request)
    assert retry.status_code == 201
    assert retry.json() == original.json()
    changed = blocked.post("/v1/uploads", headers=headers, json={**request, "size_bytes": 2})
    assert changed.status_code == 409
    fresh = blocked.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "capacity-new"},
        json={"object_key": "uploads/new.tar", "size_bytes": 1, "sha256": "3" * 64},
    )
    assert fresh.status_code == 507


def test_rolling_episode_quality_missing_and_conflict_statuses(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    url = f"/v1/datasets/pokemon/episodes/{RUN_ID}/quality-dispositions"
    body = {
        "schema_id": "nxml.episode-quality.v1",
        "training_eligible": False,
        "reason": "not_closed",
        "validator": "validator",
        "validator_version": "1",
    }
    assert (
        client.post(url, headers={"Idempotency-Key": "missing-close"}, json=body).status_code == 404
    )

    content, manifest = segment_tar(RUN_ID, 0, 0, 10)
    catalog = client.app.state.catalog
    ingest = client.app.state.ingest
    upload(catalog, ingest, client.app.state.segments, content, manifest, "rolling-quality")
    client.app.state.segments.close_episode(close_body([manifest]), idempotency_key="close")
    assert (
        client.post(url, headers={"Idempotency-Key": "quality-conflict"}, json=body).status_code
        == 201
    )
    assert (
        client.post(
            url,
            headers={"Idempotency-Key": "quality-conflict"},
            json={**body, "reason": "changed"},
        ).status_code
        == 409
    )
