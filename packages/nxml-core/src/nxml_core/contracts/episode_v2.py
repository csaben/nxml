"""Episode schema v2.

The models describe the logical Parquet rows and JSON manifest.  They do not
prescribe a storage backend or mutate the existing ``switch_packets.v1`` action
space.  Timestamp fields are integer nanoseconds from a monotonic clock; the
manifest maps that clock to UTC for correlation with other systems.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeNs = Annotated[int, Field(ge=0)]
Action = list[float]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ControllerV2(StrEnum):
    HUMAN = "human"
    POLICY = "policy"
    BLENDED = "blended"
    SAFETY = "safety"
    NONE = "none"


class ClockMappingV2(ContractModel):
    clock_id: str = Field(min_length=1)
    monotonic_origin_ns: NonNegativeNs
    utc_origin: datetime
    uncertainty_ns: NonNegativeNs = 0


class CaptureMetadataV2(ContractModel):
    host_id: str = Field(min_length=1)
    capture_device: str = Field(min_length=1)
    video_codec: str = Field(min_length=1)
    container: str = Field(min_length=1)
    pixel_format: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    nominal_fps: float = Field(gt=0)
    lossless: bool
    extra: dict[str, Any] = Field(default_factory=dict)


class FileChecksumV2(ContractModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    algorithm: Literal["sha256"] = "sha256"
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class LineageV2(ContractModel):
    capture_session_id: str = Field(min_length=1)
    parent_episode_ids: list[str] = Field(default_factory=list)
    source_dataset_ids: list[str] = Field(default_factory=list)
    source_model_id: str | None = None
    source_model_revision: str | None = None

    @model_validator(mode="after")
    def revision_requires_model(self) -> LineageV2:
        if self.source_model_revision is not None and self.source_model_id is None:
            raise ValueError("source_model_revision requires source_model_id")
        return self


class ActionRecordV2(ContractModel):
    frame_index: int = Field(ge=0)
    frame_timestamp_ns: NonNegativeNs
    action_timestamp_ns: NonNegativeNs
    action_age_ns: NonNegativeNs
    applied_action: Action
    human_action: Action | None = None
    human_action_mask: list[bool]
    policy_action: Action | None = None
    controller: ControllerV2
    ownership: list[ControllerV2]
    policy_id: str | None = None
    policy_revision: str | None = None
    valid: bool
    invalid_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def internally_consistent(self) -> ActionRecordV2:
        width = len(self.applied_action)
        for name, value in (
            ("human_action_mask", self.human_action_mask),
            ("ownership", self.ownership),
            ("human_action", self.human_action),
            ("policy_action", self.policy_action),
        ):
            if value is not None and len(value) != width:
                raise ValueError(f"{name} width {len(value)} != applied_action width {width}")
        if self.policy_revision is not None and self.policy_id is None:
            raise ValueError("policy_revision requires policy_id")
        if self.valid and self.invalid_reasons:
            raise ValueError("valid records cannot have invalid_reasons")
        if self.action_timestamp_ns > self.frame_timestamp_ns:
            raise ValueError("action_timestamp_ns cannot be after frame_timestamp_ns")
        if self.action_age_ns != self.frame_timestamp_ns - self.action_timestamp_ns:
            raise ValueError("action_age_ns must equal frame_timestamp_ns - action_timestamp_ns")
        return self


class EventRecordV2(ContractModel):
    event_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    timestamp_ns: NonNegativeNs
    frame_index: int | None = Field(default=None, ge=0)
    kind: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class EpisodeManifestV2(ContractModel):
    schema_id: Literal["nxml.episode.v2"] = "nxml.episode.v2"
    episode_id: str = Field(min_length=1)
    game_id: str = Field(min_length=1)
    config_id: str = Field(min_length=1)
    build_id: str = Field(min_length=1)
    action_spec_id: str = Field(min_length=1)
    action_dim: int = Field(gt=0)
    created_at_utc: datetime
    frame_count: int = Field(ge=0)
    first_frame_timestamp_ns: NonNegativeNs
    last_frame_timestamp_ns: NonNegativeNs
    clock_mapping: ClockMappingV2
    capture: CaptureMetadataV2
    files: list[FileChecksumV2] = Field(min_length=1)
    lineage: LineageV2

    @model_validator(mode="after")
    def valid_bounds_and_files(self) -> EpisodeManifestV2:
        if self.last_frame_timestamp_ns < self.first_frame_timestamp_ns:
            raise ValueError("last frame timestamp precedes first frame timestamp")
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest file paths must be unique")
        return self


def validate_action_records(
    records: list[ActionRecordV2], *, action_dim: int
) -> None:
    """Validate episode-level width, ordering, and monotonic timestamp invariants."""
    previous_frame_index = -1
    previous_frame_ns = -1
    previous_action_ns = -1
    for record in records:
        if len(record.applied_action) != action_dim:
            raise ValueError(
                f"frame {record.frame_index}: action width {len(record.applied_action)} != {action_dim}"
            )
        if record.frame_index <= previous_frame_index:
            raise ValueError("frame_index must be strictly increasing")
        if record.frame_timestamp_ns <= previous_frame_ns:
            raise ValueError("frame_timestamp_ns must be strictly increasing")
        if record.action_timestamp_ns < previous_action_ns:
            raise ValueError("action_timestamp_ns must be monotonic")
        previous_frame_index = record.frame_index
        previous_frame_ns = record.frame_timestamp_ns
        previous_action_ns = record.action_timestamp_ns
