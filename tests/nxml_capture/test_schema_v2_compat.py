from __future__ import annotations

from nxml_capture.schema_v2_compat import migrate_edge_action_row, migrate_edge_manifest


def test_edge_row_maps_names_without_reinterpreting_wall_clock() -> None:
    result = migrate_edge_action_row(
        {
            "frame_idx": 3,
            "timestamp": 1_700_000_000.0,
            "action": [0.0] * 26,
            "human_mask": [False] * 26,
            "ownership": [0] * 26,
            "active_driver": "neutral",
            "valid": True,
        }
    )
    assert result.compatible is False
    assert result.value["frame_timestamp_ns"] == 0
    assert result.value["action_timestamp_ns"] == 0
    assert result.value["controller"] == "none"
    assert "missing_monotonic_frame_timestamp" in result.value["invalid_reasons"]
    assert any("not converted" in note for note in result.provenance)


def test_edge_row_maps_authoritative_monotonic_fields() -> None:
    result = migrate_edge_action_row(
        {
            "frame_idx": 4,
            "frame_monotonic_ns": 200,
            "action_monotonic_ns": 150,
            "applied_action": [0.0] * 26,
            "human_action": [0.0] * 26,
            "human_mask": [False] * 26,
            "policy_action": [0.1] * 26,
            "ownership": [2] * 26,
            "active_driver": "policy",
            "policy_id": "za-ppo",
            "policy_revision": "rev-7",
            "valid": True,
        }
    )
    assert result.compatible is True
    assert result.value["frame_index"] == 4
    assert result.value["action_age_ns"] == 50
    assert result.value["human_action_mask"] == [False] * 26
    assert result.value["controller"] == "policy"


def test_action_after_frame_is_invalid_with_original_in_provenance() -> None:
    result = migrate_edge_action_row(
        {
            "frame_monotonic_ns": 100,
            "action_monotonic_ns": 120,
            "applied_action": [0.0] * 26,
            "human_mask": [False] * 26,
            "ownership": [0] * 26,
        }
    )
    assert result.compatible is False
    assert result.value["action_timestamp_ns"] == 100
    assert result.value["action_age_ns"] == 0
    assert "action_timestamp_after_frame" in result.value["invalid_reasons"]
    assert "original action_monotonic_ns=120" in result.provenance[0]


def test_manifest_mapping_reports_missing_strict_clock_provenance() -> None:
    result = migrate_edge_manifest(
        {
            "episode_id": "ep-1",
            "game": "pokemon-za",
            "action_spec": "switch_packets.v1",
            "action_dim": 26,
            "frame_count": 10,
            "created_at_utc": "2026-07-18T00:00:00+00:00",
            "clock_mapping": {
                "wall_clock": "unix_seconds",
                "monotonic_clock": "monotonic_ns",
            },
            "files": {
                "episode.mkv": {"bytes": 12, "sha256": "a" * 64},
            },
        }
    )
    assert result.compatible is False
    assert result.value["action_spec_id"] == "switch_packets.v1"
    assert result.value["files"][0]["digest"] == "a" * 64
    assert any("strict_clock_mapping" in note for note in result.provenance)
