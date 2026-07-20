"""Iterable MIRA clips streamed from immutable content-addressed WebDataset shards."""

from __future__ import annotations

import random
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch
from nxml_core.contracts import DaggerActionRecordV2
from nxml_core.contracts.webdataset_v1 import WebDatasetSnapshotV1
from torch.utils.data import IterableDataset, get_worker_info
from torchcodec.decoders import VideoDecoder

from nxwm_mira.data.batch import ClipMeta
from nxwm_mira.data.webdataset import BoundedShardCache, assigned_shards
from nxwm_mira.data.webdataset_validate import validate_segment_semantics
from nxwm_mira.data.za_dataset import pool_actions


class WebDatasetClipDataset(IterableDataset):
    """Yield the same ``(video, actions, ClipMeta)`` contract as ``ZAClipDataset``."""

    def __init__(
        self,
        manifest: WebDatasetSnapshotV1,
        *,
        cache: BoundedShardCache,
        fetch,
        split: str,
        clip_len: int,
        frame_stride: int,
        frame_hw: tuple[int, int] = (720, 1280),
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle_shards: bool = True,
    ) -> None:
        super().__init__()
        if clip_len <= 0 or frame_stride <= 0:
            raise ValueError("clip length and stride must be positive")
        self.manifest = manifest.model_copy(
            update={"shards": [item for item in manifest.shards if item.split == split]}
        )
        if not self.manifest.shards:
            raise ValueError(f"publication has no {split} shards")
        self.cache = cache
        self.fetch = fetch
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.frame_hw = frame_hw
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.epoch = 0
        episode_ids = sorted({item.episode_id for item in manifest.shards})
        self.episode_indices = {episode_id: index for index, episode_id in enumerate(episode_ids)}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        shards = assigned_shards(
            self.manifest,
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            workers=workers,
        )
        if self.shuffle_shards:
            random.Random(self.seed + self.epoch).shuffle(shards)
        for shard in shards:
            tar_path = self.cache.get(shard, self.fetch)
            validate_segment_semantics(tar_path, shard)
            with tempfile.TemporaryDirectory(prefix="nxml-wds-clip-") as raw_tmp:
                root = Path(raw_tmp)
                by_role = {member.role: member for member in shard.members}
                extracted = {}
                with tarfile.open(tar_path, "r:*") as archive:
                    for role in ("video", "actions"):
                        member = by_role[role]
                        source = archive.extractfile(member.path)
                        if source is None:
                            raise ValueError("unreadable tar member")
                        target = root / Path(member.path).name
                        with target.open("xb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
                        extracted[role] = target
                rows = pq.read_table(extracted["actions"]).to_pylist()
                actions = torch.tensor(
                    [DaggerActionRecordV2.model_validate(row).applied_action for row in rows],
                    dtype=torch.float32,
                )
                decoder = VideoDecoder(str(extracted["video"]), device="cpu")
                span = self.clip_len * self.frame_stride
                for start in range(0, len(rows) - span + 1, span):
                    indices = list(range(start, start + span, self.frame_stride))
                    video = decoder.get_frames_at(indices).data
                    if tuple(video.shape[-2:]) != self.frame_hw:
                        video = torch.nn.functional.interpolate(
                            video.float(), size=self.frame_hw, mode="bilinear", antialias=True
                        ).clamp(0, 255).to(torch.uint8)
                    pooled = pool_actions(actions, indices, self.frame_stride)
                    yield video, pooled, ClipMeta(
                        episode_index=self.episode_indices[shard.episode_id],
                        episode_id=shard.episode_id,
                        start_frame=start,
                        frame_stride=self.frame_stride,
                    )
