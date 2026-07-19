"""ZAClipDataset over a synthesized tiny corpus: shapes, stride, split, collate."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nxwm_mira.data.batch import ClipMeta, VideoBatch
from nxwm_mira.data.za_dataset import (
    ZAClipDataset,
    collate_clips,
    load_episodes,
    split_episodes,
)


def _write_episode_video(path: Path, n_frames: int, size: int = 64) -> None:
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            # Frame index encoded in brightness so stride/start assertions can read it back.
            frame_array = np.full((size, size, 3), min(i * 4, 255), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny corpus: 3 usable episodes across 2 folders + 1 degenerate 2-frame episode."""
    root = tmp_path_factory.mktemp("za-corpus")
    frame_counts = [60, 50, 700, 2]
    folders = ["folder-a", "folder-a", "folder-b", "folder-b"]
    rows = {
        "episode_index": [],
        "episode_id": [],
        "source_folder": [],
        "video_path": [],
        "frame_count": [],
    }
    for i, (n_frames, folder) in enumerate(zip(frame_counts, folders, strict=True)):
        rel = f"videos/chunk-000/episode_{i:06d}.mkv"
        _write_episode_video(root / rel, n_frames)
        rows["episode_index"].append(i)
        rows["episode_id"].append(f"ep_{i}")
        rows["source_folder"].append(folder)
        rows["video_path"].append(rel)
        rows["frame_count"].append(n_frames)
    (root / "meta").mkdir()
    pq.write_table(pa.table(rows), root / "meta" / "episodes.parquet")
    return root


def test_load_episodes(corpus: Path) -> None:
    episodes = load_episodes(corpus)
    assert len(episodes) == 4
    assert episodes[0].video_path.is_file()


def test_split_filters_short_and_holds_out_per_folder(corpus: Path) -> None:
    episodes = load_episodes(corpus)
    train, val = split_episodes(episodes, min_frames=17, holdout_per_folder=1)
    all_indices = {ep.episode_index for ep in train} | {ep.episode_index for ep in val}
    assert 3 not in all_indices  # 2-frame episode dropped
    assert len(val) == 2  # one per folder
    assert {ep.source_folder for ep in val} == {"folder-a", "folder-b"}
    assert not ({ep.episode_index for ep in train} & {ep.episode_index for ep in val})


def test_split_explicit_val_episodes(corpus: Path) -> None:
    episodes = load_episodes(corpus)
    train, val = split_episodes(episodes, min_frames=17, val_episodes=[2])
    assert [ep.episode_index for ep in val] == [2]
    assert {ep.episode_index for ep in train} == {0, 1}


def test_clip_shapes_and_stride(corpus: Path) -> None:
    episodes = [ep for ep in load_episodes(corpus) if ep.episode_index == 0]  # 60 frames
    ds = ZAClipDataset(episodes, clip_len=8, frame_stride=2, frame_hw=(64, 64))
    # span 16, non-overlapping: starts 0, 16, 32 -> 3 clips (start 48 would need frame 62)
    assert len(ds) == 3
    video, meta = ds[1]
    assert video.shape == (8, 3, 64, 64)
    assert video.dtype == torch.uint8
    assert isinstance(meta, ClipMeta)
    assert meta.start_frame == 16
    assert meta.frame_stride == 2
    # Brightness encodes the frame index (i*4): stride-2 from 16 -> frames 16,18,...
    means = video.float().mean(dim=(1, 2, 3))
    assert means[1] > means[0]  # strictly increasing brightness
    assert abs(means[0].item() - 16 * 4) < 12  # h264 is lossy; wide tolerance


def test_collate(corpus: Path) -> None:
    episodes = [ep for ep in load_episodes(corpus) if ep.episode_index == 0]
    ds = ZAClipDataset(episodes, clip_len=4, frame_hw=(64, 64))
    batch, metas = collate_clips([ds[0], ds[1]])
    assert isinstance(batch, VideoBatch)
    assert batch.video.shape == (2, 4, 3, 64, 64)
    assert len(metas) == 2


def test_mixed_resolution_episodes_collate(corpus: Path, tmp_path: Path) -> None:
    """The corpus mixes 480x640 and 720x1280 episodes; batches must still stack."""
    import shutil

    root = tmp_path / "mixed"
    shutil.copytree(corpus, root)
    rel = "videos/chunk-000/episode_000004.mkv"
    _write_episode_video(root / rel, n_frames=40, size=128)  # a differently-sized episode
    table = pq.read_table(root / "meta" / "episodes.parquet").to_pydict()
    for key, value in {
        "episode_index": 4, "episode_id": "ep_4", "source_folder": "folder-a",
        "video_path": rel, "frame_count": 40,
    }.items():
        table[key].append(value)
    pq.write_table(pa.table(table), root / "meta" / "episodes.parquet")

    episodes = load_episodes(root)
    ds = ZAClipDataset(episodes, clip_len=4, frame_hw=(64, 64))
    by_index = {ep.episode_index: ep for ep in episodes}
    small = ZAClipDataset([by_index[0]], clip_len=4, frame_hw=(64, 64))[0]
    large = ZAClipDataset([by_index[4]], clip_len=4, frame_hw=(64, 64))[0]
    batch, _ = collate_clips([small, large])
    assert batch.video.shape == (2, 4, 3, 64, 64)
    assert batch.video.dtype == torch.uint8


def test_no_clips_raises(corpus: Path) -> None:
    episodes = [ep for ep in load_episodes(corpus) if ep.episode_index == 3]  # 2 frames
    with pytest.raises(ValueError, match="No clips"):
        ZAClipDataset(episodes, clip_len=8, frame_stride=2)
