"""Packing episodes into WebDataset tar shards.

WebDataset convention: files sharing a basename prefix form one sample. Each episode
contributes three members named ``{episode_id}.{mkv|mp4}``, ``{episode_id}.parquet``,
``{episode_id}.json`` (the manifest). Shards are plain uncompressed tars (the video inside
is already compressed) built in a staging dir, with a ``.json`` sidecar listing contents.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from nxml_spool.episodes import Episode


@dataclass(frozen=True)
class Shard:
    path: Path  # shard-{NNNNNN}.tar in the staging dir
    sidecar_path: Path  # shard-{NNNNNN}.json
    episode_ids: list[str]
    size_bytes: int
    sha256: str
    api_manifest: dict[str, object]


def pack_shard(
    episodes: list[Episode], staging_dir: Path, shard_index: int
) -> Shard:
    """Write ``shard-{index:06d}.tar`` + sidecar from the given episodes."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_path = staging_dir / f"shard-{shard_index:06d}.tar"
    tmp_path = shard_path.with_suffix(".tar.tmp")

    members: list[dict[str, object]] = []

    def add_member(tar: tarfile.TarFile, source: Path, arcname: str, kind: str) -> None:
        tar.add(source, arcname=arcname)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        members.append(
            {
                "path": arcname,
                "kind": kind,
                "size_bytes": source.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )

    with tarfile.open(tmp_path, "w") as tar:
        for ep in episodes:
            add_member(
                tar,
                ep.video_path,
                f"{ep.episode_id}{ep.video_path.suffix}",
                "video",
            )
            add_member(tar, ep.parquet_path, f"{ep.episode_id}.parquet", "actions")
            if ep.events_path is not None:
                add_member(
                    tar,
                    ep.events_path,
                    f"{ep.episode_id}.events.parquet",
                    "events",
                )
            add_member(tar, ep.manifest_path, f"{ep.episode_id}.json", "episode_manifest")
    tmp_path.replace(shard_path)
    digest = hashlib.sha256()
    with shard_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    shard_sha256 = digest.hexdigest()

    api_manifest: dict[str, object] = {
        "schema_id": "nxml.episode.v2",
        "action_spec_id": "switch_packets.v1",
        "episodes": [{"episode_id": ep.episode_id} for ep in episodes],
        "members": members,
    }
    sidecar = {
        "shard": shard_path.name,
        "episodes": [
            {
                "episode_id": ep.episode_id,
                "video": ep.video_path.name,
                "bytes": ep.size_bytes,
            }
            for ep in episodes
        ],
        "n_episodes": len(episodes),
        "bytes": shard_path.stat().st_size,
        "sha256": shard_sha256,
        "api_manifest": api_manifest,
    }
    sidecar_path = shard_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return Shard(
        path=shard_path,
        sidecar_path=sidecar_path,
        episode_ids=[ep.episode_id for ep in episodes],
        size_bytes=shard_path.stat().st_size,
        sha256=shard_sha256,
        api_manifest=api_manifest,
    )


def plan_shards(episodes: list[Episode], *, shard_size_bytes: int) -> list[list[Episode]]:
    """Greedy split of episodes (already ordered) into shard-sized groups.

    A single episode larger than the target still gets its own shard. The final
    partial group is returned too — the caller decides whether to pack it now
    (flush/final mode) or wait for more episodes.
    """
    groups: list[list[Episode]] = []
    current: list[Episode] = []
    current_bytes = 0
    for ep in episodes:
        size = ep.size_bytes
        if current and current_bytes + size > shard_size_bytes:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(ep)
        current_bytes += size
    if current:
        groups.append(current)
    return groups
