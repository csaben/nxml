"""Immutable Hugging Face WebDataset publication contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = r"^[0-9a-f]{64}$"
CONTENT_ID = r"^sha256:[0-9a-f]{64}$"


class PublishedMemberV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["video", "actions", "events"]
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256)


class CodecLineageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    codec: Literal["h264", "ffv1"]
    container: Literal["matroska", "mov,mp4,m4a,3gp,3g2,mj2"]
    profile: str = Field(min_length=1)
    level: int | None = Field(default=None, ge=0)
    pixel_format: Literal["yuv420p", "bgr0"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    nominal_fps: float = Field(gt=0)
    time_base: str = Field(pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    gop_size: int = Field(gt=0)
    aspect_mode: Literal["pad"] = "pad"


class PublishedShardV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(pattern=r"^shards/[0-9a-f]{2}/[0-9a-f]{64}\.tar$")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256)
    episode_id: str = Field(min_length=1)
    segment_id: str = Field(pattern=CONTENT_ID)
    sequence_index: int = Field(ge=0)
    split: Literal["train", "validation", "canary"]
    codec: CodecLineageV1
    members: list[PublishedMemberV1] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_identity(self):
        if self.segment_id != "sha256:" + self.sha256:
            raise ValueError("segment identity must match shard checksum")
        if self.path != f"shards/{self.sha256[:2]}/{self.sha256}.tar":
            raise ValueError("shard path must be content addressed")
        roles = [member.role for member in self.members]
        if sorted(roles) != ["actions", "events", "video"] or len(set(roles)) != 3:
            raise ValueError("shard requires one video/actions/events member")
        if len({member.path for member in self.members}) != 3:
            raise ValueError("member paths must be unique")
        return self


class WebDatasetSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.hf-webdataset-snapshot.v1"]
    dataset_id: str = Field(min_length=1)
    source_snapshot_id: str = Field(pattern=CONTENT_ID)
    action_spec_id: Literal["switch_packets.v1"]
    publication_kind: Literal["canary", "snapshot"]
    shards: list[PublishedShardV1] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self):
        identities = [(item.episode_id, item.sequence_index) for item in self.shards]
        if identities != sorted(identities):
            raise ValueError("shards must be ordered by episode and segment sequence")
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate episode segment")
        by_episode: dict[str, list[int]] = {}
        for episode_id, sequence_index in identities:
            by_episode.setdefault(episode_id, []).append(sequence_index)
        if any(indices != list(range(len(indices))) for indices in by_episode.values()):
            raise ValueError("episode segment sequence must start at zero without gaps")
        if self.publication_kind == "canary" and any(s.split != "canary" for s in self.shards):
            raise ValueError("canary publication may contain only canary shards")
        return self
