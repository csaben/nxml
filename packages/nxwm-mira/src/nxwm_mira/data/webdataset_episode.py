"""Deterministic episode-level assignment and MIRA window planning."""

from __future__ import annotations

from dataclasses import dataclass

from nxml_core.contracts.webdataset_v2 import WebDatasetSnapshotV2


@dataclass(frozen=True)
class SegmentSpan:
    segment_id: str
    local_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class EpisodeWindow:
    episode_id: str
    global_frame_indices: tuple[int, ...]
    spans: tuple[SegmentSpan, ...]


def assigned_episode_ids(
    manifest: WebDatasetSnapshotV2,
    *,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    workers_per_rank: int = 1,
) -> tuple[str, ...]:
    """Assign whole episodes without duplication across ranks or workers."""
    if world_size < 1 or workers_per_rank < 1:
        raise ValueError("world_size and workers_per_rank must be positive")
    if not 0 <= rank < world_size or not 0 <= worker_id < workers_per_rank:
        raise ValueError("rank or worker_id out of range")
    consumer = rank * workers_per_rank + worker_id
    consumer_count = world_size * workers_per_rank
    episode_ids = tuple(episode.episode_id for episode in manifest.episodes)
    return episode_ids[consumer::consumer_count]


def iter_episode_windows(
    manifest: WebDatasetSnapshotV2,
    *,
    split: str,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    workers_per_rank: int = 1,
) -> tuple[EpisodeWindow, ...]:
    """Plan fixed MIRA windows, including windows crossing shard boundaries."""
    assigned = set(
        assigned_episode_ids(
            manifest,
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            workers_per_rank=workers_per_rank,
        )
    )
    contract = manifest.training.window
    by_episode = {
        episode.episode_id: [
            segment for segment in manifest.segments if segment.episode_id == episode.episode_id
        ]
        for episode in manifest.episodes
    }
    windows: list[EpisodeWindow] = []
    for episode in manifest.episodes:
        if episode.episode_id not in assigned or episode.split != split:
            continue
        segments = by_episode[episode.episode_id]
        first = segments[0].temporal.frame_index_origin
        stop = segments[-1].temporal.frame_index_origin + segments[-1].temporal.frame_count
        source_span = (contract.clip_frames - 1) * contract.frame_stride + 1
        for start in range(first, stop - source_span + 1, contract.frame_stride):
            indices = tuple(
                start + offset * contract.frame_stride for offset in range(contract.clip_frames)
            )
            spans: list[SegmentSpan] = []
            for segment in segments:
                segment_start = segment.temporal.frame_index_origin
                segment_stop = segment_start + segment.temporal.frame_count
                local = [index - segment_start for index in indices if segment_start <= index < segment_stop]
                if local:
                    spans.append(
                        SegmentSpan(
                            segment_id=segment.segment_id,
                            local_frame_indices=tuple(local),
                        )
                    )
            if sum(len(span.local_frame_indices) for span in spans) != contract.clip_frames:
                raise ValueError("window cannot be reconstructed from ordered segments")
            windows.append(EpisodeWindow(episode.episode_id, indices, tuple(spans)))
    return tuple(windows)
