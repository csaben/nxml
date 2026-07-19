from datetime import UTC, datetime

import pytest
from nxml_core.contracts.episode_v2 import (
    ActionRecordV2,
    CaptureMetadataV2,
    ClockMappingV2,
    ControllerV2,
    EpisodeManifestV2,
    FileChecksumV2,
    LineageV2,
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
        ownership=[ControllerV2.HUMAN] * 26,
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
