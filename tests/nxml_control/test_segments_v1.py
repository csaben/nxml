import hashlib
import io
import tarfile

import pytest
from nxml_control.catalog import Catalog
from nxml_control.segments import SegmentCatalog
from nxml_control.service import IngestService
from nxml_control.storage import LocalObjectStorage


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
        object_key=f"segments/{manifest['object_sha256']}.tar",
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
    content0, manifest0 = segment_tar("run", 0, 100, 200)
    pending = catalog.create_upload(
        idempotency_key="segment-0",
        object_key=f"segments/{manifest0['object_sha256']}.tar",
        size_bytes=len(content0),
        sha256=manifest0["object_sha256"],
    )
    ingest.upload(pending.id, io.BytesIO(content0))
    assert segments.status("pokemon")["committed_segments"] == 0

    # Process restart after durable bytes but before registration.
    segments = SegmentCatalog(catalog, ingest, ingest.storage)
    receipt0 = segments.commit_segment(pending.id, manifest0)
    assert segments.commit_segment(pending.id, manifest0) == receipt0

    content1, manifest1 = segment_tar("run", 1, 200, 300)
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
    assert [item[1] for item in reconstructed] == [content0, content1]
    status = segments.status("pokemon")
    assert status["committed_segments"] == 2
    assert status["committed_bytes"] == len(content0) + len(content1)
    assert status["closed_episodes"] == status["eligible_episodes"] == 1


def test_episode_close_rejects_gap_overlap_missing_and_digest_mismatch(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    _, first = segment_tar("bad", 0, 0, 10)
    _, second = segment_tar("bad", 1, 10, 20)
    for index, manifest in enumerate((first, second)):
        content, _ = segment_tar("bad", index, index * 10, (index + 1) * 10)
        upload(catalog, ingest, segments, content, manifest, f"bad-{index}")

    missing = close_body([first, second])
    missing["segments"][1]["segment_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="missing segment"):
        segments.close_episode(missing, idempotency_key="missing")

    digest = close_body([first, second])
    digest["segments"][1]["object_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="metadata mismatch"):
        segments.close_episode(digest, idempotency_key="digest")

    gap = close_body([first, second])
    gap["segments"][1]["timeline_start_ns"] = 11
    with pytest.raises(ValueError, match="gap"):
        segments.close_episode(gap, idempotency_key="gap")

    overlap = close_body([first, second])
    overlap["segments"][1]["timeline_start_ns"] = 9
    with pytest.raises(ValueError, match="overlap"):
        segments.close_episode(overlap, idempotency_key="overlap")

    order = close_body([first, second])
    order["segments"][1]["sequence_index"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        segments.close_episode(order, idempotency_key="order")


def test_segment_quality_propagates_to_snapshot_without_mutating_receipt(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content, manifest = segment_tar("quality", 0, 0, 100)
    receipt = upload(catalog, ingest, segments, content, manifest, "quality-segment")
    close = segments.close_episode(close_body([manifest]), idempotency_key="quality-close")
    disposition = segments.set_quality(
        manifest["segment_id"],
        idempotency_key="quality-v1",
        training_eligible=False,
        reason="noncausal_action_alignment",
        validator="alignment",
        validator_version="1",
    )
    assert (
        segments.set_quality(
            manifest["segment_id"],
            idempotency_key="quality-v1",
            training_eligible=False,
            reason="noncausal_action_alignment",
            validator="alignment",
            validator_version="1",
        )["disposition_id"]
        == disposition["disposition_id"]
    )
    snapshot = segments.create_snapshot("pokemon")
    assert snapshot["episodes"] == []
    exclusion = snapshot["excluded_episodes"][0]
    assert exclusion["episode_id"] == "quality"
    assert exclusion["segments"][0]["reason"] == "noncausal_action_alignment"
    assert segments.get_receipt(receipt.receipt_id) == receipt
    assert next(segments.iter_reconstruction(close["close_id"]))[1] == content

    segments.set_quality(
        manifest["segment_id"],
        idempotency_key="quality-v2",
        training_eligible=True,
        reason="alignment_corrected",
        validator="alignment",
        validator_version="2",
    )
    assert segments.get_snapshot(snapshot["snapshot_id"]) == snapshot
    replacement = segments.create_snapshot("pokemon")
    assert replacement["snapshot_id"] != snapshot["snapshot_id"]
    assert replacement["episodes"][0]["episode_id"] == "quality"


def test_segment_member_rejects_unsafe_path(tmp_path):
    catalog, ingest, segments = system(tmp_path)
    content, manifest = segment_tar("unsafe", 0, 0, 10)
    manifest["members"][0]["path"] = "../events.parquet"
    pending = catalog.create_upload(
        idempotency_key="unsafe",
        object_key=f"segments/{manifest['object_sha256']}.tar",
        size_bytes=len(content),
        sha256=manifest["object_sha256"],
    )
    ingest.upload(pending.id, io.BytesIO(content))
    with pytest.raises(ValueError, match="safe relative path"):
        segments.commit_segment(pending.id, manifest)
