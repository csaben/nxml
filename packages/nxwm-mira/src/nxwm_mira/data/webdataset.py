"""Strict content-addressed WebDataset access with a bounded restart-safe cache."""

from __future__ import annotations

import hashlib
import os
import tarfile
from collections.abc import Callable, Iterator
from pathlib import Path

from nxml_core.contracts.webdataset_v1 import PublishedShardV1, WebDatasetSnapshotV1

CHUNK_BYTES = 1024 * 1024


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class BoundedShardCache:
    """Download immutable shards atomically and evict oldest unused entries."""

    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("cache max_bytes must be positive")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        for partial in self.root.glob("*.partial"):
            partial.unlink()

    def get(self, shard: PublishedShardV1, fetch: Callable[[str, Path], None]) -> Path:
        target = self.root / f"{shard.sha256}.tar"
        if target.exists():
            if file_digest(target) == (shard.size_bytes, shard.sha256):
                os.utime(target, None)
                return target
            target.unlink()
        partial = target.with_suffix(".partial")
        partial.unlink(missing_ok=True)
        fetch(shard.path, partial)
        if file_digest(partial) != (shard.size_bytes, shard.sha256):
            partial.unlink(missing_ok=True)
            raise ValueError(f"shard checksum mismatch: {shard.path}")
        partial.replace(target)
        self._evict(protected=target)
        return target

    def _evict(self, *, protected: Path) -> None:
        entries = sorted(
            (path for path in self.root.glob("*.tar") if path != protected),
            key=lambda path: (path.stat().st_atime_ns, path.name),
        )
        total = sum(path.stat().st_size for path in self.root.glob("*.tar"))
        for path in entries:
            if total <= self.max_bytes:
                break
            size = path.stat().st_size
            path.unlink()
            total -= size
        if protected.stat().st_size > self.max_bytes:
            protected.unlink()
            raise ValueError("one shard exceeds bounded cache capacity")


def load_publication(path: str | Path) -> WebDatasetSnapshotV1:
    return WebDatasetSnapshotV1.model_validate_json(Path(path).read_text())


def assigned_shards(
    manifest: WebDatasetSnapshotV1, *, rank: int, world_size: int, worker_id: int, workers: int
) -> list[PublishedShardV1]:
    if world_size <= 0 or workers <= 0:
        raise ValueError("worker counts must be positive")
    if rank < 0 or rank >= world_size or worker_id < 0 or worker_id >= workers:
        raise ValueError("invalid distributed worker coordinates")
    global_worker = rank * workers + worker_id
    total_workers = world_size * workers
    return [
        shard
        for index, shard in enumerate(manifest.shards)
        if index % total_workers == global_worker
    ]


def verify_shard(path: Path, shard: PublishedShardV1) -> None:
    if file_digest(path) != (shard.size_bytes, shard.sha256):
        raise ValueError("shard checksum mismatch")
    declared = {member.path: member for member in shard.members}
    with tarfile.open(path, "r:*") as archive:
        infos = archive.getmembers()
        if len(infos) != 3 or {item.name for item in infos} != set(declared):
            raise ValueError("tar member set mismatch")
        for info in infos:
            if not info.isfile() or Path(info.name).is_absolute() or ".." in Path(info.name).parts:
                raise ValueError("unsafe tar member")
            stream = archive.extractfile(info)
            if stream is None:
                raise ValueError("unreadable tar member")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(CHUNK_BYTES):
                size += len(chunk)
                digest.update(chunk)
            member = declared[info.name]
            if (size, digest.hexdigest()) != (member.size_bytes, member.sha256):
                raise ValueError(f"member checksum mismatch: {info.name}")


def iter_verified_shards(
    manifest: WebDatasetSnapshotV1,
    cache: BoundedShardCache,
    fetch: Callable[[str, Path], None],
    *,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    workers: int = 1,
) -> Iterator[tuple[PublishedShardV1, Path]]:
    for shard in assigned_shards(
        manifest, rank=rank, world_size=world_size, worker_id=worker_id, workers=workers
    ):
        path = cache.get(shard, fetch)
        verify_shard(path, shard)
        yield shard, path


def hub_fetcher(*, repo_id: str, revision: str, xet_cache: str | Path):
    """Return a fetch callback pinned to one immutable Hub revision."""
    os.environ.setdefault("HF_XET_CACHE", str(Path(xet_cache)))

    def fetch(remote_path: str, destination: Path) -> None:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(repo_id, remote_path, repo_type="dataset", revision=revision)
        )
        with downloaded.open("rb") as source, destination.open("xb") as target:
            while chunk := source.read(CHUNK_BYTES):
                target.write(chunk)

    return fetch
