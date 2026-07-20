"""Complete MIRA-ready immutable WebDataset snapshot contract."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nxml_core.contracts.webdataset_v1 import PublishedMemberV1

SHA256 = r"^[0-9a-f]{64}$"
CONTENT_ID = r"^sha256:[0-9a-f]{64}$"


class CompactCodecLineageV2(BaseModel):
    """Artifact-derived compact ingest properties required by MIRA."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    compatibility_id: Literal["nxml.compact-h264-main32-720p60.v1"]
    codec: Literal["h264"]
    container: Literal["matroska"]
    profile: Literal["Main"]
    level: Literal[32]
    pixel_format: Literal["yuv420p"]
    width: Literal[1280]
    height: Literal[720]
    r_frame_rate: str = Field(pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    avg_frame_rate: str = Field(pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    time_base: str = Field(pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    gop_size: Literal[60]
    max_b_frames: Literal[0]
    measured_bit_rate: int = Field(gt=0)
    aspect_mode: Literal["pad"]
    artifact_probe: Literal["ffprobe"]

    @model_validator(mode="after")
    def validate_main_level_32_limits(self):
        max_fs = 5_120
        max_mbps = 216_000
        max_br_bits_per_second = 20_000_000
        macroblocks_per_frame = math.ceil(self.width / 16) * math.ceil(self.height / 16)
        rate = Fraction(self.r_frame_rate)
        if rate != Fraction(self.avg_frame_rate):
            raise ValueError("nominal and average frame rates must match")
        macroblocks_per_second = macroblocks_per_frame * rate
        if macroblocks_per_frame > max_fs:
            raise ValueError("frame size exceeds H.264 Level 3.2 MaxFS")
        if macroblocks_per_second > max_mbps:
            raise ValueError("frame rate exceeds H.264 Level 3.2 MaxMBPS")
        if rate != 60:
            raise ValueError("compact profile requires exact 60 fps")
        if self.measured_bit_rate > max_br_bits_per_second:
            raise ValueError("measured bitrate exceeds Main Level 3.2 MaxBR")
        return self


class TemporalMappingV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    clock_id: str = Field(min_length=1)
    timeline_start_ns: int = Field(ge=0)
    timeline_end_ns: int = Field(gt=0)
    frame_index_origin: int = Field(ge=0)
    frame_count: int = Field(gt=0)
    video_frame_mapping: Literal["decoded_ordinal_equals_frame_idx"]
    frame_timestamp_column: Literal["frame_monotonic_ns"]
    action_timestamp_column: Literal["action_monotonic_ns"]
    interval_semantics: Literal["half_open"] = "half_open"

    @model_validator(mode="after")
    def validate_interval(self):
        if self.timeline_end_ns <= self.timeline_start_ns:
            raise ValueError("temporal interval must be non-empty")
        return self


class ControlModelLineageV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["human", "hybrid", "pure_ai"]
    action_schema_id: Literal["nxml.dagger-actions.v2"]
    action_spec_id: Literal["switch_packets.v1"]
    mute_mask_version: Literal["switch_packets.v1/mute.v1"]
    policy_id: str | None = None
    policy_revision: str | None = None
    policy_digest: str | None = Field(default=None, pattern=CONTENT_ID)

    @model_validator(mode="after")
    def validate_policy_identity(self):
        values = (self.policy_id, self.policy_revision, self.policy_digest)
        if self.mode == "human" and any(value is not None for value in values):
            raise ValueError("human lineage cannot claim a policy")
        if self.mode != "human" and any(value is None for value in values):
            raise ValueError("AI lineage requires immutable policy id/revision/digest")
        return self


class PublishedSegmentV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(pattern=r"^shards/[0-9a-f]{2}/[0-9a-f]{64}\.tar$")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256)
    episode_id: str = Field(min_length=1)
    segment_id: str = Field(pattern=CONTENT_ID)
    sequence_index: int = Field(ge=0)
    split: Literal["train", "validation", "canary"]
    temporal: TemporalMappingV2
    codec: CompactCodecLineageV2
    control: ControlModelLineageV2
    members: list[PublishedMemberV1] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_identity(self):
        if self.segment_id != "sha256:" + self.sha256:
            raise ValueError("segment identity must match shard checksum")
        if self.path != f"shards/{self.sha256[:2]}/{self.sha256}.tar":
            raise ValueError("shard path must be content addressed")
        if sorted(member.role for member in self.members) != ["actions", "events", "video"]:
            raise ValueError("segment requires exact video/actions/events triplet")
        if len({member.path for member in self.members}) != 3:
            raise ValueError("segment member paths must be unique")
        return self


class PublishedEpisodeV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_id: str = Field(min_length=1)
    close_id: str = Field(pattern=CONTENT_ID)
    split: Literal["train", "validation", "canary"]
    clock_id: str = Field(min_length=1)
    timeline_start_ns: int = Field(ge=0)
    timeline_end_ns: int = Field(gt=0)
    segment_ids: list[str] = Field(min_length=1)


class MiraWindowContractV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_fps: float = Field(gt=0)
    target_fps: float = Field(gt=0)
    frame_stride: int = Field(gt=0)
    clip_frames: int = Field(gt=0)
    action_pooling: Literal["sticks_mean_buttons_or"]
    aspect_mode: Literal["pad"]
    frame_size_hw: tuple[int, int]

    @model_validator(mode="after")
    def validate_fps(self):
        if abs(self.source_fps / self.target_fps - self.frame_stride) > 1e-9:
            raise ValueError("fps ratio must equal integer frame_stride")
        return self


class ImmutableTrainingLineageV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    publisher_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_snapshot_id: str = Field(pattern=CONTENT_ID)
    hub_repo_id: str = Field(min_length=1)
    hub_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    window: MiraWindowContractV2
    checkpoint_mode: Literal["finetune_from", "continue_from", "none"]
    checkpoint_id: str | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self):
        if (self.checkpoint_mode == "none") != (self.checkpoint_id is None):
            raise ValueError("checkpoint mode and immutable checkpoint id disagree")
        return self


class WebDatasetSnapshotV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.hf-webdataset-snapshot.v2"]
    dataset_id: str = Field(min_length=1)
    source_snapshot_id: str = Field(pattern=CONTENT_ID)
    publication_kind: Literal["canary", "snapshot"]
    episodes: list[PublishedEpisodeV2] = Field(min_length=1)
    segments: list[PublishedSegmentV2] = Field(min_length=1)
    training: ImmutableTrainingLineageV2

    @model_validator(mode="after")
    def validate_graph(self):
        if self.training.source_snapshot_id != self.source_snapshot_id:
            raise ValueError("training lineage snapshot mismatch")
        if self.publication_kind == "canary" and any(
            segment.split != "canary" for segment in self.segments
        ):
            raise ValueError("canary publication may contain only canary segments")
        if [(s.episode_id, s.sequence_index) for s in self.segments] != sorted(
            (s.episode_id, s.sequence_index) for s in self.segments
        ):
            raise ValueError("segments must be ordered by episode and sequence")
        by_episode: dict[str, list[PublishedSegmentV2]] = {}
        for segment in self.segments:
            by_episode.setdefault(segment.episode_id, []).append(segment)
        if set(by_episode) != {episode.episode_id for episode in self.episodes}:
            raise ValueError("episode and segment identity sets differ")
        for episode in self.episodes:
            segments = by_episode[episode.episode_id]
            if [item.sequence_index for item in segments] != list(range(len(segments))):
                raise ValueError("episode segment sequence must start at zero without gaps")
            if [item.segment_id for item in segments] != episode.segment_ids:
                raise ValueError("episode segment identity/order mismatch")
            if any(item.split != episode.split for item in segments):
                raise ValueError("split must be episode-stable")
            if any(item.temporal.clock_id != episode.clock_id for item in segments):
                raise ValueError("clock must be episode-stable")
            if segments[0].temporal.timeline_start_ns != episode.timeline_start_ns:
                raise ValueError("episode start does not match first segment")
            if segments[-1].temporal.timeline_end_ns != episode.timeline_end_ns:
                raise ValueError("episode end does not match last segment")
            for left, right in itertools.pairwise(segments):
                if left.temporal.timeline_end_ns != right.temporal.timeline_start_ns:
                    raise ValueError("segment timeline gap or overlap")
                if (
                    left.temporal.frame_index_origin + left.temporal.frame_count
                    != right.temporal.frame_index_origin
                ):
                    raise ValueError("global frame mapping gap or overlap")
        return self
