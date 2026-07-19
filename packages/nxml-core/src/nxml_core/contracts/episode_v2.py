"""Episode schema v2.

The models describe the logical Parquet rows and JSON manifest.  They do not
prescribe a storage backend or mutate the existing ``switch_packets.v1`` action
space.  Timestamp fields are integer nanoseconds from a monotonic clock; the
manifest maps that clock to UTC for correlation with other systems.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import IntEnum, StrEnum
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


class OwnershipCodeV2(IntEnum):
    """Canonical Parquet wire encoding: 0=unowned, 1=human, 2=policy."""

    UNOWNED = 0
    HUMAN = 1
    POLICY = 2


class DaggerModeV2(StrEnum):
    HUMAN_ONLY = "human_only"
    POLICY_ONLY = "policy_only"
    HYBRID = "hybrid"


class OwnershipSourceV2(StrEnum):
    NEUTRAL = "neutral"
    HUMAN_TAKEOVER = "human_takeover"
    PER_DIMENSION = "per_dimension"
    POLICY = "policy"


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
    ownership: list[OwnershipCodeV2]
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


class DaggerActionRecordV2(ContractModel):
    """Strict physical Parquet row for DAgger provenance."""

    row_schema_id: Literal["nxml.dagger-actions.v2"] = "nxml.dagger-actions.v2"
    action_spec_id: Literal["switch_packets.v1"] = "switch_packets.v1"
    frame_idx: int = Field(ge=0)
    frame_monotonic_ns: NonNegativeNs
    policy_action: Action | None
    policy_source_frame_monotonic_ns: NonNegativeNs | None
    policy_cluster_proposal_monotonic_ns: NonNegativeNs | None
    policy_action_monotonic_ns: NonNegativeNs | None
    policy_action_age_ns: NonNegativeNs | None
    policy_action_valid: bool
    policy_action_fresh: bool
    human_action: Action | None
    human_action_monotonic_ns: NonNegativeNs | None
    human_action_age_ns: NonNegativeNs | None
    human_action_valid: bool
    human_action_fresh: bool
    muted_policy_action: Action | None
    muted_policy_action_monotonic_ns: NonNegativeNs | None
    applied_action: Action
    applied_action_monotonic_ns: NonNegativeNs
    applied_action_age_ns: NonNegativeNs
    applied_action_valid: bool
    human_mask: list[bool]
    mute_mask: list[bool]
    mute_mask_version: Literal["switch_packets.v1"]
    ownership: list[OwnershipCodeV2]
    ownership_source: OwnershipSourceV2
    mode: DaggerModeV2
    takeover_active: bool
    policy_revision_id: str | None
    policy_checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bc_training_eligible: bool
    valid: bool
    invalid_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def faithful_provenance(self) -> DaggerActionRecordV2:
        packets = {
            "applied_action": self.applied_action,
            "human_mask": self.human_mask,
            "mute_mask": self.mute_mask,
            "ownership": self.ownership,
        }
        for name, value in (
            ("policy_action", self.policy_action),
            ("human_action", self.human_action),
            ("muted_policy_action", self.muted_policy_action),
        ):
            if value is not None:
                packets[name] = value
        for name, value in packets.items():
            if len(value) != 26:
                raise ValueError(f"{name} must have 26 dimensions")
        for name in ("applied_action", "policy_action", "human_action", "muted_policy_action"):
            value = getattr(self, name)
            if value is not None and not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain only finite values")

        timed_sources = (
            ("policy_action", self.policy_action_monotonic_ns, self.policy_action_age_ns),
            ("human_action", self.human_action_monotonic_ns, self.human_action_age_ns),
            (
                "muted_policy_action",
                self.muted_policy_action_monotonic_ns,
                self.policy_action_age_ns,
            ),
            ("applied_action", self.applied_action_monotonic_ns, self.applied_action_age_ns),
        )
        for name, timestamp, age in timed_sources:
            packet = getattr(self, name)
            if packet is None:
                if timestamp is not None or (name != "muted_policy_action" and age is not None):
                    raise ValueError(f"absent {name} cannot have timestamp/age")
                continue
            if timestamp is None or age is None:
                raise ValueError(f"{name} requires timestamp and age")
            if timestamp > self.frame_monotonic_ns:
                raise ValueError(f"{name} timestamp cannot follow frame")
            if age != self.frame_monotonic_ns - timestamp:
                raise ValueError(f"{name} age must equal frame minus source timestamp")

        has_policy = self.policy_action is not None
        if has_policy != (self.muted_policy_action is not None):
            raise ValueError("policy_action and muted_policy_action must be present together")
        if has_policy != (self.policy_revision_id is not None):
            raise ValueError("policy action requires immutable revision ID")
        if has_policy != (self.policy_checkpoint_sha256 is not None):
            raise ValueError("policy action requires checkpoint SHA-256")
        if has_policy != (self.policy_source_frame_monotonic_ns is not None):
            raise ValueError("policy action requires its edge source-frame timestamp")
        if has_policy != (self.policy_cluster_proposal_monotonic_ns is not None):
            raise ValueError("policy action requires its cluster proposal timestamp")
        if (
            self.policy_source_frame_monotonic_ns is not None
            and self.policy_source_frame_monotonic_ns > self.frame_monotonic_ns
        ):
            raise ValueError("policy source frame cannot follow captured frame")
        if has_policy and self.muted_policy_action_monotonic_ns != self.policy_action_monotonic_ns:
            raise ValueError("muted policy packet must preserve proposal timestamp")
        if self.policy_action_valid and not has_policy:
            raise ValueError("valid policy action is missing")
        if self.human_action_valid and self.human_action is None:
            raise ValueError("valid human action is missing")
        if self.policy_action_fresh and not self.policy_action_valid:
            raise ValueError("fresh policy action must be valid")
        if self.human_action_fresh and not self.human_action_valid:
            raise ValueError("fresh human action must be valid")

        if has_policy:
            for index, muted in enumerate(self.mute_mask):
                expected = 0.0 if muted else float(self.policy_action[index])
                if float(self.muted_policy_action[index]) != expected:
                    raise ValueError("muted_policy_action does not match mute_mask")
        elif any(self.mute_mask):
            raise ValueError("mute mask cannot be active without a policy proposal")

        if self.mode == DaggerModeV2.HUMAN_ONLY and has_policy:
            raise ValueError("human_only rows cannot contain a policy proposal")
        if self.mode in {DaggerModeV2.POLICY_ONLY, DaggerModeV2.HYBRID} and not has_policy:
            raise ValueError("policy mode requires a policy proposal")
        if self.mode == DaggerModeV2.POLICY_ONLY and OwnershipCodeV2.HUMAN in self.ownership:
            raise ValueError("policy_only rows cannot be human-owned")
        if self.takeover_active:
            if self.mode != DaggerModeV2.HYBRID:
                raise ValueError("takeover is only valid in hybrid mode")
            if self.ownership_source != OwnershipSourceV2.HUMAN_TAKEOVER:
                raise ValueError("takeover requires human_takeover ownership source")
            if any(owner != OwnershipCodeV2.HUMAN for owner in self.ownership):
                raise ValueError("human takeover must mark every dimension human-owned")

        for index, owner in enumerate(self.ownership):
            if owner == OwnershipCodeV2.HUMAN:
                if not (self.human_action_valid and self.human_action_fresh):
                    raise ValueError("human-owned dimensions require a valid fresh human packet")
                expected = self.human_action[index]
            elif owner == OwnershipCodeV2.POLICY:
                if not (self.policy_action_valid and self.policy_action_fresh):
                    raise ValueError("policy-owned dimensions require a valid fresh policy packet")
                expected = self.muted_policy_action[index]
            else:
                expected = 0.0
            if float(self.applied_action[index]) != float(expected):
                raise ValueError("applied_action does not match per-dimension ownership")

        if self.valid and self.invalid_reasons:
            raise ValueError("valid records cannot have invalid_reasons")
        if self.bc_training_eligible and not (
            self.valid
            and self.applied_action_valid
            and OwnershipCodeV2.HUMAN in self.ownership
        ):
            raise ValueError("BC eligibility requires valid applied action with human ownership")
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


def validate_action_records(records: list[ActionRecordV2], *, action_dim: int) -> None:
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
