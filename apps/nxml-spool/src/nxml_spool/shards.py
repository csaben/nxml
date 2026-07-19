"""Packing episodes into WebDataset tar shards.

WebDataset convention: files sharing a basename prefix form one sample. Each episode
contributes three members named ``{episode_id}.{mkv|mp4}``, ``{episode_id}.parquet``,
``{episode_id}.json`` (the manifest). Shards are plain uncompressed tars (the video inside
is already compressed) built in a staging dir, with a ``.json`` sidecar listing contents.
"""

from __future__ import annotations

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


def pack_shard(
    episodes: list[Episode], staging_dir: Path, shard_index: int
) -> Shard:
    """Write ``shard-{index:06d}.tar`` + sidecar from the given episodes."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    shard_path = staging_dir / f"shard-{shard_index:06d}.tar"
    tmp_path = shard_path.with_suffix(".tar.tmp")

    with tarfile.open(tmp_path, "w") as tar:
        for ep in episodes:
            tar.add(ep.video_path, arcname=f"{ep.episode_id}{ep.video_path.suffix}")
            tar.add(ep.parquet_path, arcname=f"{ep.episode_id}.parquet")
            if ep.events_path is not None:
                tar.add(ep.events_path, arcname=f"{ep.episode_id}.events.parquet")
            tar.add(ep.manifest_path, arcname=f"{ep.episode_id}.json")
    tmp_path.replace(shard_path)

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
    }
    sidecar_path = shard_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return Shard(
        path=shard_path,
        sidecar_path=sidecar_path,
        episode_ids=[ep.episode_id for ep in episodes],
        size_bytes=shard_path.stat().st_size,
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
