from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from nxml_capture import SyncedFrame, VideoParquetEpisodeWriter


def test_schema_v2_preserves_proposals_ownership_and_checksums(tmp_path: Path) -> None:
    writer = VideoParquetEpisodeWriter(
        tmp_path,
        episode_name="episode",
        codec="h264",
        fps=30,
        game="pokemon-za",
        lineage={"parent_policy_revision": "rev-6"},
    )
    human = np.zeros(26, dtype=np.float32)
    human[25] = 1
    human_mask = human.astype(bool)
    policy = np.linspace(-1, 1, 26, dtype=np.float32)
    applied = policy.copy()
    applied[human_mask] = human[human_mask]
    ownership = np.full(26, 2, dtype=np.uint8)
    ownership[human_mask] = 1
    writer.append(
        SyncedFrame(
            timestamp=1000.25,
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            action=applied,
            action_age=0.012,
            frame_monotonic_ns=5_000_000_000,
            action_timestamp=1000.238,
            action_monotonic_ns=4_988_000_000,
            human_action=human,
            human_mask=human_mask,
            policy_action=policy,
            ownership=ownership,
            controller_id="web:gamepad-0",
            active_driver="human+policy",
            policy_id="za-ppo",
            policy_revision="rev-7",
            policy_digest="sha256:abc",
            human_monotonic_ns=4_987_000_000,
            policy_monotonic_ns=4_986_000_000,
            policy_observation_monotonic_ns=4_980_000_000,
            muted_policy_action=policy,
            mute_mask=np.zeros(26, dtype=bool),
            mute_mask_version="switch_packets.v1/mute.v1",
            ownership_source="human",
            mode="hybrid",
            takeover=True,
            proposal_valid=True,
            proposal_fresh=True,
        )
    )
    writer.append_event(
        "driver_changed",
        timestamp=1000.2,
        monotonic_ns=4_950_000_000,
        payload={"driver": "human+policy"},
    )
    assert writer.close() is not None

    table = pq.read_table(tmp_path / "episode.parquet")
    row = table.to_pylist()[0]
    assert row["action"] == row["applied_action"]
    assert row["human_action"][25] == 1
    assert row["human_mask"][25] is True
    assert row["ownership"][25] == 1
    assert row["ownership"][0] == 2
    assert row["policy_revision"] == "rev-7"
    assert row["frame_monotonic_ns"] == 5_000_000_000
    assert row["frame_timestamp_ns"] == 5_000_000_000
    assert row["action_timestamp_ns"] == 4_988_000_000
    assert row["action_age_ns"] == 12_000_000
    assert row["policy_digest"] == "sha256:abc"
    assert row["mode"] == "hybrid" and row["takeover"] is True
    assert row["proposal_valid"] is True and row["proposal_fresh"] is True
    assert row["bc_training_eligible"] is False  # blended ownership is not BC ground truth.

    events = pq.read_table(tmp_path / "episode.events.parquet").to_pylist()
    assert events[0]["kind"] == "driver_changed"

    manifest = json.loads((tmp_path / "episode.manifest.json").read_text())
    assert manifest["episode_id"] == writer.episode_id
    assert manifest["schema_id"] == "nxml.episode.v2"
    assert manifest["action_spec"] == "switch_packets.v1"
    assert manifest["action_spec_id"] == "switch_packets.v1"
    assert manifest["action_schema_id"] == "nxml.dagger-actions.v2"
    assert manifest["action_schema_version"] == 2
    assert manifest["action_rows_schema_id"] == "nxml.dagger-actions.v2"
    assert manifest["lineage"]["parent_policy_revision"] == "rev-6"
    assert manifest["first_frame_timestamp_ns"] == 5_000_000_000
    assert manifest["last_frame_timestamp_ns"] == 5_000_000_000
    assert manifest["clock_mapping"]["clock_id"] == "linux-monotonic"
    assert manifest["clock_mapping"]["monotonic_origin_ns"] > 0
    for name, metadata in manifest["files"].items():
        payload = (tmp_path / name).read_bytes()
        assert metadata["bytes"] == len(payload)
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()


def test_schema_v2_rejects_future_or_inconsistent_valid_action(tmp_path: Path) -> None:
    writer = VideoParquetEpisodeWriter(tmp_path, episode_name="bad", codec="h264", fps=30)
    base = dict(
        timestamp=1.0,
        frame=np.zeros((4, 4, 3), np.uint8),
        action=np.zeros(26, np.float32),
        frame_monotonic_ns=100,
        action_monotonic_ns=101,
        action_age=-1e-9,
    )
    with pytest.raises(ValueError, match="cannot follow"):
        writer.append(SyncedFrame(**base))
    with pytest.raises(ValueError, match="sparse invalid reason"):
        writer.append(SyncedFrame(**{**base, "valid": False}))
