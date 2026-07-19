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
    HUMAN = "human"
    PURE_AI = "pure_ai"
    HYBRID = "hybrid"


class OwnershipSourceV2(StrEnum):
    NONE = "none"
    HUMAN = "human"
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
    """Strict superset of logical v2 and the deployed physical Parquet row."""

    row_schema_id: Literal["nxml.dagger-actions.v2"] | None = None
    action_spec_id: Literal["switch_packets.v1"] | None = None
    frame_index: int | None = Field(default=None, ge=0)
    frame_idx: int | None = Field(default=None, ge=0)
    timestamp: float | None = None
    frame_timestamp_ns: NonNegativeNs | None = None
    frame_monotonic_ns: NonNegativeNs | None = None
    action_timestamp: float | None = None
    action_timestamp_ns: NonNegativeNs | None = None
    action_monotonic_ns: NonNegativeNs | None = None
    action_age_ns: NonNegativeNs | None = None
    action_age: float | None = Field(default=None, ge=0)
    action: Action | None = None
    applied_action: Action
    human_action: Action | None = None
    human_action_mask: list[bool] | None = None
    human_mask: list[bool] | None = None
    policy_action: Action | None = None
    muted_policy_action: Action | None = None
    mute_mask: list[bool] | None = None
    mute_mask_version: str | None = None
    controller: ControllerV2 | str | None = None
    controller_id: str | None = None
    active_driver: str | None = None
    ownership: list[OwnershipCodeV2]
    ownership_source: str | None = None
    mode: str | None = None
    takeover: bool = False
    policy_id: str | None = None
    policy_revision: str | None = None
    policy_digest: str | None = None
    human_monotonic_ns: NonNegativeNs | None = None
    policy_monotonic_ns: NonNegativeNs | None = None
    policy_observation_monotonic_ns: NonNegativeNs | None = None
    proposal_valid: bool = False
    proposal_fresh: bool = False
    applied_action_valid: bool | None = None
    bc_training_eligible: bool = False
    valid: bool
    invalid_reasons: list[str] = Field(default_factory=list)

    @property
    def effective_frame_index(self) -> int:
        value = self.frame_idx if self.frame_idx is not None else self.frame_index
        assert value is not None
        return value

    @property
    def effective_frame_ns(self) -> int | None:
        return (
            self.frame_monotonic_ns
            if self.frame_monotonic_ns is not None
            else self.frame_timestamp_ns
        )

    @property
    def effective_action_ns(self) -> int | None:
        return (
            self.action_monotonic_ns
            if self.action_monotonic_ns is not None
            else self.action_timestamp_ns
        )

    @model_validator(mode="after")
    def internally_consistent(self) -> ActionRecordV2:
        if self.frame_index is None and self.frame_idx is None:
            raise ValueError("frame_index or frame_idx is required")
        for left, right, name in (
            (self.frame_index, self.frame_idx, "frame index aliases"),
            (self.frame_timestamp_ns, self.frame_monotonic_ns, "frame timestamp aliases"),
            (self.action_timestamp_ns, self.action_monotonic_ns, "action timestamp aliases"),
        ):
            if left is not None and right is not None and left != right:
                raise ValueError(f"{name} must match")

        packets = {
            "applied_action": self.applied_action,
            "ownership": self.ownership,
        }
        for name in (
            "action",
            "human_action",
            "human_action_mask",
            "human_mask",
            "policy_action",
            "muted_policy_action",
            "mute_mask",
        ):
            value = getattr(self, name)
            if value is not None:
                packets[name] = value
        width = len(self.applied_action)
        for name, value in packets.items():
            if len(value) != width:
                raise ValueError(f"{name} width {len(value)} != applied_action width {width}")
        for name in ("action", "applied_action", "human_action", "policy_action", "muted_policy_action"):
            value = getattr(self, name)
            if value is not None and not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain only finite values")
        if self.action is not None and self.action != self.applied_action:
            raise ValueError("action compatibility alias must equal applied_action")
        if (
            self.human_action_mask is not None
            and self.human_mask is not None
            and self.human_action_mask != self.human_mask
        ):
            raise ValueError("human mask aliases must match")
        if self.policy_revision is not None and self.policy_id is None:
            raise ValueError("policy_revision requires policy_id")
        if self.proposal_fresh and not self.proposal_valid:
            raise ValueError("fresh policy proposal must be valid")

        frame_ns, action_ns = self.effective_frame_ns, self.effective_action_ns
        physical_dagger = self.frame_idx is not None
        applied_valid = self.valid if self.applied_action_valid is None else self.applied_action_valid
        if frame_ns is None:
            raise ValueError("records require a frame monotonic timestamp")
        if self.valid:
            if self.invalid_reasons:
                raise ValueError("valid records cannot have invalid_reasons")
            if frame_ns is None or action_ns is None or self.action_age_ns is None:
                raise ValueError("valid records require frame/action monotonic timestamps and age")
            if action_ns > frame_ns:
                raise ValueError("action timestamp cannot be after frame timestamp")
            if self.action_age_ns != frame_ns - action_ns:
                raise ValueError("action_age_ns must equal frame timestamp minus action timestamp")
            if self.action_age is not None and not math.isclose(
                self.action_age, self.action_age_ns / 1_000_000_000, rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError("action_age must match action_age_ns")
            if self.policy_action is not None and self.muted_policy_action is not None:
                mask = self.mute_mask or [False] * width
                for index, muted in enumerate(mask):
                    expected = 0.0 if muted else float(self.policy_action[index])
                    if float(self.muted_policy_action[index]) != expected:
                        raise ValueError("muted_policy_action does not match mute_mask")
            if self.takeover and (
                self.mode != "hybrid"
                or any(owner != OwnershipCodeV2.HUMAN for owner in self.ownership)
            ):
                raise ValueError("takeover requires hybrid mode with every dimension human-owned")
            for index, owner in enumerate(self.ownership) if physical_dagger else ():
                if owner == OwnershipCodeV2.HUMAN:
                    if self.human_action is None or (
                        physical_dagger and self.human_monotonic_ns is None
                    ):
                        raise ValueError("human-owned dimensions require a timestamped human packet")
                    expected = self.human_action[index]
                elif owner == OwnershipCodeV2.POLICY:
                    policy_ready = self.muted_policy_action is not None
                    if physical_dagger:
                        policy_ready = policy_ready and all(
                            (
                                self.proposal_valid,
                                self.proposal_fresh,
                                self.policy_monotonic_ns is not None,
                                self.policy_revision is not None,
                                self.policy_digest is not None,
                            )
                        )
                    if not policy_ready:
                        raise ValueError(
                            "policy-owned dimensions require a fresh immutable policy proposal"
                        )
                    expected = self.muted_policy_action[index]
                else:
                    expected = 0.0
                if float(self.applied_action[index]) != float(expected):
                    raise ValueError("applied_action does not match per-dimension ownership")
        else:
            if not self.invalid_reasons:
                raise ValueError("invalid records require invalid_reasons")
            if any(float(value) != 0.0 for value in self.applied_action):
                raise ValueError("invalid records must have neutral applied_action")
            if self.action is not None and any(float(value) != 0.0 for value in self.action):
                raise ValueError("invalid records must have neutral action")
            for name in ("human_action", "policy_action", "muted_policy_action"):
                packet = getattr(self, name)
                if packet is not None and any(float(value) != 0.0 for value in packet):
                    raise ValueError(f"invalid records must have neutral {name}")
            for name in ("human_action_mask", "human_mask", "mute_mask"):
                mask = getattr(self, name)
                if mask is not None and any(mask):
                    raise ValueError(f"invalid records must have a clear {name}")
            if any(owner != OwnershipCodeV2.UNOWNED for owner in self.ownership):
                raise ValueError("invalid records must be unowned")
            if self.proposal_valid or self.proposal_fresh or self.takeover:
                raise ValueError("invalid records cannot claim proposal/takeover state")
            if applied_valid:
                raise ValueError("invalid records cannot claim a valid applied action")
            if self.bc_training_eligible:
                raise ValueError("invalid records cannot be BC training eligible")

        if self.bc_training_eligible and not (
            self.valid and applied_valid and OwnershipCodeV2.HUMAN in self.ownership
        ):
            raise ValueError("BC eligibility requires a valid human-owned applied action")
        return self


class DaggerActionRecordV2(ActionRecordV2):
    """Named strict validator for deployed DAgger Parquet rows."""


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
        frame_index = record.effective_frame_index
        frame_ns = record.effective_frame_ns
        action_ns = record.effective_action_ns
        assert frame_ns is not None
        if len(record.applied_action) != action_dim:
            raise ValueError(
                f"frame {frame_index}: action width {len(record.applied_action)} != {action_dim}"
            )
        if frame_index <= previous_frame_index:
            raise ValueError("frame_index must be strictly increasing")
        if frame_ns <= previous_frame_ns:
            raise ValueError("frame_timestamp_ns must be strictly increasing")
        if action_ns is not None and action_ns < previous_action_ns:
            raise ValueError("action_timestamp_ns must be monotonic")
        previous_frame_index = frame_index
        previous_frame_ns = frame_ns
        if action_ns is not None:
            previous_action_ns = action_ns
