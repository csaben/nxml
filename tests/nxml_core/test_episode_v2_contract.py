from datetime import UTC, datetime

import pytest
from nxml_core.contracts.episode_v2 import (
    ActionRecordV2,
    CaptureMetadataV2,
    ClockMappingV2,
    ControllerV2,
    DaggerActionRecordV2,
    EpisodeManifestV2,
    FileChecksumV2,
    LineageV2,
    OwnershipCodeV2,
    validate_action_records,
)
from pydantic import ValidationError


def _record(index: int, *, frame_ns: int, action_ns: int) -> ActionRecordV2:
    return ActionRecordV2(
        frame_index=index,
        frame_timestamp_ns=frame_ns,
        action_timestamp_ns=action_ns,
        action_age_ns=frame_ns - action_ns,
        applied_action=[0.0] * 26,
        human_action=[0.0] * 26,
        human_action_mask=[True] * 26,
        policy_action=None,
        controller=ControllerV2.HUMAN,
        ownership=[OwnershipCodeV2.HUMAN] * 26,
        valid=True,
    )


def test_manifest_preserves_existing_action_spec_identifier() -> None:
    manifest = EpisodeManifestV2(
        episode_id="ep-1",
        game_id="pokemon-za",
        config_id="capture-default",
        build_id="git:abc",
        action_spec_id="switch_packets.v1",
        action_dim=26,
        created_at_utc=datetime.now(UTC),
        frame_count=2,
        first_frame_timestamp_ns=100,
        last_frame_timestamp_ns=200,
        clock_mapping=ClockMappingV2(
            clock_id="linux-monotonic", monotonic_origin_ns=10, utc_origin=datetime.now(UTC)
        ),
        capture=CaptureMetadataV2(
            host_id="edge-1",
            capture_device="elgato",
            video_codec="ffv1",
            container="mkv",
            pixel_format="bgr0",
            width=1920,
            height=1080,
            nominal_fps=30,
            lossless=True,
        ),
        files=[FileChecksumV2(path="ep-1.tar", size_bytes=12, digest="a" * 64)],
        lineage=LineageV2(capture_session_id="session-1"),
    )
    assert manifest.schema_id == "nxml.episode.v2"
    assert manifest.action_spec_id == "switch_packets.v1"
    assert EpisodeManifestV2.model_validate_json(manifest.model_dump_json()) == manifest


def test_action_record_requires_matching_widths() -> None:
    payload = _record(0, frame_ns=200, action_ns=150).model_dump()
    payload["human_action_mask"] = [False] * 25
    with pytest.raises(ValidationError, match="human_action_mask width"):
        ActionRecordV2.model_validate(payload)


def test_action_record_requires_exact_age() -> None:
    payload = _record(0, frame_ns=200, action_ns=150).model_dump()
    payload["action_age_ns"] = 1
    with pytest.raises(ValidationError, match="action_age_ns"):
        ActionRecordV2.model_validate(payload)


def test_episode_validation_rejects_non_monotonic_rows() -> None:
    records = [_record(0, frame_ns=100, action_ns=90), _record(1, frame_ns=99, action_ns=91)]
    with pytest.raises(ValueError, match="frame_timestamp_ns"):
        validate_action_records(records, action_dim=26)


def test_contract_forbids_unknown_fields() -> None:
    payload = _record(0, frame_ns=100, action_ns=90).model_dump()
    payload["legacy_action"] = [0.0] * 26
    with pytest.raises(ValidationError, match="legacy_action"):
        ActionRecordV2.model_validate(payload)


def test_ownership_uses_canonical_parquet_integer_codes() -> None:
    payload = _record(0, frame_ns=100, action_ns=90).model_dump(mode="json")
    assert payload["ownership"] == [1] * 26
    payload["ownership"][0:3] = [0, 1, 2]
    assert ActionRecordV2.model_validate(payload).ownership[:3] == [
        OwnershipCodeV2.UNOWNED,
        OwnershipCodeV2.HUMAN,
        OwnershipCodeV2.POLICY,
    ]


def _dagger_row(**changes) -> dict:
    human = [0.0] * 26
    human[0] = 1.0
    policy = [0.25] * 26
    muted = policy.copy()
    muted[1] = 0.0
    ownership = [1] * 26
    payload = {
        "row_schema_id": "nxml.dagger-actions.v2",
        "action_spec_id": "switch_packets.v1",
        "frame_idx": 0,
        "frame_monotonic_ns": 1_000,
        "policy_action": policy,
        "policy_source_frame_monotonic_ns": 850,
        "policy_cluster_proposal_monotonic_ns": 123_456,
        "policy_action_monotonic_ns": 900,
        "policy_action_age_ns": 100,
        "policy_action_valid": True,
        "policy_action_fresh": True,
        "human_action": human,
        "human_action_monotonic_ns": 950,
        "human_action_age_ns": 50,
        "human_action_valid": True,
        "human_action_fresh": True,
        "muted_policy_action": muted,
        "muted_policy_action_monotonic_ns": 900,
        "applied_action": human,
        "applied_action_monotonic_ns": 950,
        "applied_action_age_ns": 50,
        "applied_action_valid": True,
        "human_mask": [True] + [False] * 25,
        "mute_mask": [False, True] + [False] * 24,
        "mute_mask_version": "switch_packets.v1",
        "ownership": ownership,
        "ownership_source": "human_takeover",
        "mode": "hybrid",
        "takeover_active": True,
        "policy_revision_id": "2f772838-2179-4739-a53c-aa95dcea35a0",
        "policy_checkpoint_sha256": "3" * 64,
        "bc_training_eligible": True,
        "valid": True,
        "invalid_reasons": [],
    }
    payload.update(changes)
    return payload


def test_dagger_row_preserves_proposal_mute_applied_and_takeover_provenance() -> None:
    parsed = DaggerActionRecordV2.model_validate(_dagger_row())
    assert parsed.ownership == [OwnershipCodeV2.HUMAN] * 26
    assert parsed.muted_policy_action[1] == 0.0
    assert parsed.applied_action == parsed.human_action


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"policy_checkpoint_sha256": "4" * 63}, "policy_checkpoint_sha256"),
        ({"policy_action_age_ns": 99}, "age must equal"),
        ({"ownership": [1] * 25 + [2]}, "takeover must mark every"),
        ({"applied_action": [0.0] * 26}, "does not match per-dimension ownership"),
        ({"bc_training_eligible": True, "applied_action_valid": False}, "BC eligibility"),
    ],
)
def test_dagger_row_rejects_unfaithful_provenance(change, message) -> None:
    with pytest.raises(ValidationError, match=message):
        DaggerActionRecordV2.model_validate(_dagger_row(**change))
