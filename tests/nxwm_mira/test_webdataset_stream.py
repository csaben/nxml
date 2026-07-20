import hashlib
import io
import tarfile
from pathlib import Path

import pytest
from nxml_core.contracts.webdataset_v1 import WebDatasetSnapshotV1
from nxwm_mira.data.webdataset import (
    BoundedShardCache,
    assigned_shards,
    iter_verified_shards,
    verify_shard,
)
from pydantic import ValidationError


def make_tar(tmp_path: Path, episode: str, sequence: int, marker: bytes = b"video"):
    prefix = f"{episode}.{sequence:06d}"
    members = {
        f"{prefix}.mkv": marker,
        f"{prefix}.parquet": b"actions",
        f"{prefix}.events.parquet": b"events",
    }
    path = tmp_path / f"source-{episode}-{sequence}.tar"
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    declared = []
    for name, data in members.items():
        role = "video" if name.endswith(".mkv") else (
            "events" if name.endswith(".events.parquet") else "actions"
        )
        declared.append(
            {"role": role, "path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    return path, {
        "path": f"shards/{digest[:2]}/{digest}.tar",
        "size_bytes": len(content),
        "sha256": digest,
        "episode_id": episode,
        "segment_id": "sha256:" + digest,
        "sequence_index": sequence,
        "split": "canary",
        "codec": {
            "codec": "h264", "container": "matroska", "profile": "High",
            "level": 42, "pixel_format": "yuv420p", "width": 1280, "height": 720,
            "nominal_fps": 60.0, "time_base": "1/1000", "gop_size": 60,
            "aspect_mode": "pad",
        },
        "members": declared,
    }


def publication(shards):
    return WebDatasetSnapshotV1.model_validate(
        {
            "schema_id": "nxml.hf-webdataset-snapshot.v1",
            "dataset_id": "nxml-pokemon-za-v2",
            "source_snapshot_id": "sha256:" + "a" * 64,
            "action_spec_id": "switch_packets.v1",
            "publication_kind": "canary",
            "shards": shards,
        }
    )


def test_manifest_fails_closed_on_schema_order_and_codec(tmp_path):
    _path0, shard0 = make_tar(tmp_path, "episode", 0)
    _path1, shard1 = make_tar(tmp_path, "episode", 1, b"video-1")
    assert len(publication([shard0, shard1]).shards) == 2
    with pytest.raises(ValidationError, match="schema_id"):
        publication([{**shard0, "schema_id": "unknown"}])
    with pytest.raises(ValidationError, match="without gaps"):
        publication([{**shard1, "sequence_index": 2}])
    bad_codec = {**shard0, "codec": {**shard0["codec"], "codec": "hevc"}}
    with pytest.raises(ValidationError, match="codec"):
        publication([bad_codec])


def test_member_and_outer_checksums_fail_closed(tmp_path):
    path, shard = make_tar(tmp_path, "episode", 0)
    parsed = publication([shard]).shards[0]
    verify_shard(path, parsed)
    corrupt = tmp_path / "corrupt.tar"
    corrupt.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="shard checksum"):
        verify_shard(corrupt, parsed)
    bad_member = parsed.model_copy(
        update={"members": [parsed.members[0].model_copy(update={"sha256": "0" * 64}), *parsed.members[1:]]}
    )
    with pytest.raises(ValueError, match="member checksum"):
        verify_shard(path, bad_member)


def test_distributed_assignment_is_deterministic_and_duplicate_free(tmp_path):
    shards = [make_tar(tmp_path, f"episode-{index}", 0, str(index).encode())[1] for index in range(8)]
    manifest = publication(shards)
    assignments = [
        assigned_shards(manifest, rank=rank, world_size=2, worker_id=worker, workers=2)
        for rank in range(2)
        for worker in range(2)
    ]
    identities = [item.segment_id for group in assignments for item in group]
    assert len(identities) == len(set(identities)) == 8
    assert assignments == [
        assigned_shards(manifest, rank=rank, world_size=2, worker_id=worker, workers=2)
        for rank in range(2)
        for worker in range(2)
    ]


def test_cache_eviction_restart_recovery_and_missing_fetch(tmp_path):
    source0, shard0 = make_tar(tmp_path, "episode-0", 0)
    source1, shard1 = make_tar(tmp_path, "episode-1", 0, b"larger-video")
    manifest = publication([shard0, shard1])
    sources = {shard0["path"]: source0, shard1["path"]: source1}

    def fetch(remote, destination):
        destination.write_bytes(sources[remote].read_bytes())

    cache_root = tmp_path / "cache"
    (cache_root).mkdir()
    (cache_root / "crash.partial").write_bytes(b"partial")
    cache = BoundedShardCache(cache_root, max_bytes=max(shard0["size_bytes"], shard1["size_bytes"]) + 1)
    assert not (cache_root / "crash.partial").exists()
    seen = [item.segment_id for item, _ in iter_verified_shards(manifest, cache, fetch)]
    assert seen == [shard0["segment_id"], shard1["segment_id"]]
    assert len(list(cache_root.glob("*.tar"))) == 1
    cache = BoundedShardCache(cache_root, max_bytes=100_000)
    assert cache.get(manifest.shards[1], fetch).exists()
    with pytest.raises(KeyError):
        cache.get(manifest.shards[0], lambda remote, destination: (_ for _ in ()).throw(KeyError(remote)))
