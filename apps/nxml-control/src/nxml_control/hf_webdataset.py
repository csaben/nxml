"""Publish immutable cluster segment snapshots as content-addressed WebDataset shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nxml_core.contracts.webdataset_v1 import (
    CodecLineageV1,
    PublishedMemberV1,
    PublishedShardV1,
    WebDatasetSnapshotV1,
)

from nxml_control.catalog import Catalog
from nxml_control.storage import LocalObjectStorage

CHUNK_BYTES = 1024 * 1024


def _copy_verified(source, destination: Path, *, size_bytes: int, sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source, partial.open("xb") as target:
        while chunk := source.read(CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
            target.write(chunk)
    if (size, digest.hexdigest()) != (size_bytes, sha256):
        partial.unlink(missing_ok=True)
        raise ValueError("cluster shard checksum mismatch")
    partial.replace(destination)


def stage_segment_snapshot(
    *,
    state_dir: str | Path,
    snapshot_id: str,
    output_dir: str | Path,
    codec_by_episode: dict[str, dict[str, Any]],
    split_by_episode: dict[str, str],
    max_shards: int,
) -> tuple[WebDatasetSnapshotV1, Path]:
    """Stage at most ``max_shards`` objects; never mutate or delete cluster storage."""
    if max_shards <= 0 or max_shards > 99:
        raise ValueError("publication batch must contain 1..99 shards")
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    catalog = Catalog(state_dir / "catalog.sqlite3")
    storage = LocalObjectStorage(state_dir / "objects")
    with catalog.connect() as db:
        row = db.execute(
            "SELECT manifest_json FROM segment_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
    if row is None:
        raise KeyError(snapshot_id)
    source_snapshot = json.loads(row["manifest_json"])
    if source_snapshot.get("schema_id") != "nxml.segment-snapshot.v1":
        raise ValueError("unsupported source snapshot schema")

    staged: list[PublishedShardV1] = []
    for episode in source_snapshot["episodes"]:
        episode_id = episode["episode_id"]
        try:
            codec = CodecLineageV1.model_validate(codec_by_episode[episode_id])
            split = split_by_episode[episode_id]
        except KeyError as error:
            raise ValueError(f"missing immutable codec/split lineage for {episode_id}") from error
        for expected_sequence, segment_id in enumerate(episode["segment_ids"]):
            if len(staged) >= max_shards:
                break
            with catalog.connect() as db:
                segment = db.execute(
                    "SELECT * FROM segment_bundles WHERE segment_id=?", (segment_id,)
                ).fetchone()
            if segment is None:
                raise ValueError(f"snapshot segment missing: {segment_id}")
            manifest = json.loads(segment["manifest_json"])
            if (
                manifest["episode_id"] != episode_id
                or manifest["sequence_index"] != expected_sequence
                or manifest["object_sha256"] != segment["sha256"]
            ):
                raise ValueError("snapshot segment ordering/identity mismatch")
            digest = segment["sha256"]
            remote_path = f"shards/{digest[:2]}/{digest}.tar"
            destination = output_dir / remote_path
            if destination.exists():
                actual = hashlib.sha256()
                actual_size = 0
                with destination.open("rb") as source:
                    while chunk := source.read(CHUNK_BYTES):
                        actual_size += len(chunk)
                        actual.update(chunk)
                if (actual_size, actual.hexdigest()) != (segment["size_bytes"], digest):
                    raise ValueError("staged shard conflicts with immutable identity")
            else:
                _copy_verified(
                    storage.open(segment["storage_key"]),
                    destination,
                    size_bytes=segment["size_bytes"],
                    sha256=digest,
                )
            staged.append(
                PublishedShardV1(
                    path=remote_path,
                    size_bytes=segment["size_bytes"],
                    sha256=digest,
                    episode_id=episode_id,
                    segment_id=segment_id,
                    sequence_index=expected_sequence,
                    split=split,
                    codec=codec,
                    members=[
                        PublishedMemberV1.model_validate(item) for item in manifest["members"]
                    ],
                )
            )
        if len(staged) >= max_shards:
            break
    if not staged:
        raise ValueError("snapshot has no stageable shards")
    publication = WebDatasetSnapshotV1(
        schema_id="nxml.hf-webdataset-snapshot.v1",
        dataset_id=source_snapshot["dataset_id"],
        source_snapshot_id=snapshot_id,
        action_spec_id="switch_packets.v1",
        publication_kind="snapshot",
        shards=staged,
    )
    manifest_path = output_dir / "snapshots" / snapshot_id.removeprefix("sha256:") / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(publication.model_dump_json(indent=2) + "\n")
    return publication, manifest_path


def publish_batch(
    *, repo_id: str, root: str | Path, manifest: WebDatasetSnapshotV1, manifest_path: Path
) -> str:
    """Upload one bounded shard batch plus its manifest in a single Hub commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(root)
    operations = [
        CommitOperationAdd(path_in_repo=item.path, path_or_fileobj=root / item.path)
        for item in manifest.shards
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=str(manifest_path.relative_to(root)), path_or_fileobj=manifest_path
        )
    )
    if len(operations) > 100:
        raise ValueError("Hub commit exceeds bounded 100-file batch")
    result = HfApi().create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"publish {manifest.publication_kind} {manifest.source_snapshot_id}",
    )
    return result.oid
