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
        "frame_idx": 0,
        "timestamp": 1000.25,
        "frame_monotonic_ns": 5_000_000_000,
        "frame_timestamp_ns": 5_000_000_000,
        "action_timestamp": 1000.238,
        "action_monotonic_ns": 4_988_000_000,
        "action_timestamp_ns": 4_988_000_000,
        "action_age_ns": 12_000_000,
        "action_age": 0.012,
        "action": human,
        "applied_action": human,
        "policy_action": policy,
        "human_action": human,
        "muted_policy_action": muted,
        "human_mask": [True] + [False] * 25,
        "mute_mask": [False, True] + [False] * 24,
        "mute_mask_version": "switch_packets.v1/mute.v1",
        "ownership": ownership,
        "controller_id": "dagger-arbitrator:switch_packets.v1",
        "active_driver": "human",
        "controller": "human",
        "ownership_source": "human",
        "mode": "hybrid",
        "takeover": True,
        "policy_id": "bc_transformer_v1",
        "policy_revision": "2f772838-2179-4739-a53c-aa95dcea35a0",
        "policy_digest": "336036bbf2d35ce7ffc303a242c9f561ba537093f85ed4e1b4ce63a4dbc9c038",
        "human_monotonic_ns": 4_987_000_000,
        "policy_monotonic_ns": 4_986_000_000,
        "policy_observation_monotonic_ns": 4_980_000_000,
        "proposal_valid": True,
        "proposal_fresh": True,
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
    assert parsed.model_dump(mode="json", exclude_unset=True) == _dagger_row()


def test_deployed_edge_row_without_eligibility_round_trips_fail_closed() -> None:
    live_writer_row = _dagger_row()
    live_writer_row.pop("bc_training_eligible")
    parsed = DaggerActionRecordV2.model_validate(live_writer_row)
    assert parsed.bc_training_eligible is False
    assert parsed.model_dump(mode="json", exclude_unset=True) == live_writer_row


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"action_age_ns": 99}, "action_age_ns must equal"),
        ({"action": [0.0] * 26}, "alias must equal"),
        (
            {
                "bc_training_eligible": True,
                "ownership": [2] * 26,
                "takeover": False,
                "applied_action": [0.25, 0.0, *([0.25] * 24)],
                "action": [0.25, 0.0, *([0.25] * 24)],
            },
            "BC eligibility",
        ),
    ],
)
def test_dagger_row_rejects_unfaithful_provenance(change, message) -> None:
    with pytest.raises(ValidationError, match=message):
        DaggerActionRecordV2.model_validate(_dagger_row(**change))


def test_invalid_neutral_edge_row_allows_null_action_time_and_fails_closed() -> None:
    zero = [0.0] * 26
    invalid = _dagger_row(
        action=zero,
        applied_action=zero,
        human_action=zero,
        policy_action=zero,
        muted_policy_action=zero,
        ownership=[0] * 26,
        human_mask=[False] * 26,
        mute_mask=[False] * 26,
        action_timestamp=None,
        action_monotonic_ns=None,
        action_timestamp_ns=None,
        action_age_ns=None,
        action_age=0.0,
        controller_id=None,
        active_driver="none",
        controller="none",
        ownership_source=None,
        mode=None,
        takeover=False,
        policy_id=None,
        policy_revision=None,
        policy_digest=None,
        human_monotonic_ns=None,
        policy_monotonic_ns=None,
        policy_observation_monotonic_ns=None,
        proposal_valid=False,
        proposal_fresh=False,
        bc_training_eligible=False,
        valid=False,
        invalid_reasons=["no_prior_arbitration_record"],
    )
    parsed = ActionRecordV2.model_validate(invalid)
    assert parsed.effective_action_ns is None
    assert parsed.bc_training_eligible is False


def test_invalid_row_rejects_non_neutral_or_missing_reason() -> None:
    with pytest.raises(ValidationError, match="neutral applied_action"):
        ActionRecordV2.model_validate(
            _dagger_row(valid=False, invalid_reasons=["missing"], bc_training_eligible=False)
        )
    with pytest.raises(ValidationError, match="require invalid_reasons"):
        ActionRecordV2.model_validate(
            _dagger_row(
                valid=False,
                invalid_reasons=[],
                applied_action=[0.0] * 26,
                action=[0.0] * 26,
                ownership=[0] * 26,
                bc_training_eligible=False,
            )
        )
