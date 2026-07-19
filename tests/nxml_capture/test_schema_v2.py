from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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

    events = pq.read_table(tmp_path / "episode.events.parquet").to_pylist()
    assert events[0]["kind"] == "driver_changed"

    manifest = json.loads((tmp_path / "episode.manifest.json").read_text())
    assert manifest["schema_id"] == "nxml.episode.v2"
    assert manifest["action_spec"] == "switch_packets.v1"
    assert manifest["lineage"]["parent_policy_revision"] == "rev-6"
    for name, metadata in manifest["files"].items():
        payload = (tmp_path / name).read_bytes()
        assert metadata["bytes"] == len(payload)
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
