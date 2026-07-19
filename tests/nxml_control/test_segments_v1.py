import hashlib
import io
import tarfile

import pytest
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
        object_key=(
            f"uploads/segments/{RUN_ID}/000000-{manifest0['object_sha256']}.tar"
        ),
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
    assert b"".join(
        chunk for _, chunk in segments.iter_reconstruction(close["close_id"], chunk_size=7)
    ) == content

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
        object_key=(
            f"uploads/segments/{UNSAFE_ID}/000000-{manifest['object_sha256']}.tar"
        ),
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


def test_inactive_service_boundary_declares_auditable_read_routes():
    assert ("GET", "/v1/segments/{segment_id}/quality-dispositions") in SEGMENT_ROUTE_CONTRACTS
    assert ("POST", "/v1/datasets/{dataset_id}/segment-snapshots") in SEGMENT_ROUTE_CONTRACTS
    assert ("GET", "/v1/segment-snapshots/{snapshot_id}") in SEGMENT_ROUTE_CONTRACTS
