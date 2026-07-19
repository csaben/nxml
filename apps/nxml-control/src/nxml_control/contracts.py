"""Strict control-plane request contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemberV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1)
    kind: Literal["video", "actions", "events", "episode_manifest", "other"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ShardEpisodeV2(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    episode_id: str = Field(min_length=1)


class ShardManifestV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.episode.v2"]
    action_spec_id: Literal["switch_packets.v1"]
    episodes: list[ShardEpisodeV2] = Field(min_length=1)
    members: list[MemberV2] = Field(min_length=1)

    @model_validator(mode="after")
    def complete(self):
        paths = [item.path for item in self.members]
        if len(paths) != len(set(paths)):
            raise ValueError("member paths must be unique")
        ids = [item.episode_id for item in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("episode IDs must be unique")
        if not any(
            item.kind == "events" and item.path.endswith(".events.parquet") for item in self.members
        ):
            raise ValueError("manifest requires a checksummed .events.parquet member")
        if not any(
            item.kind == "actions" and item.path.endswith(".parquet") for item in self.members
        ):
            raise ValueError("manifest requires a checksummed actions parquet member")
        return self
