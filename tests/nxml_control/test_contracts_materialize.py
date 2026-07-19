import pytest
from nxml_control.contracts import ShardManifestV2
from nxml_control.materialize import select_training_rows
from pydantic import ValidationError


def manifest():
    return {
        "schema_id": "nxml.episode.v2",
        "action_spec_id": "switch_packets.v1",
        "episodes": [{"episode_id": "ep-1"}],
        "members": [
            {"path": "ep-1.parquet", "kind": "actions", "size_bytes": 1, "sha256": "a" * 64},
            {"path": "ep-1.events.parquet", "kind": "events", "size_bytes": 1, "sha256": "b" * 64},
            {"path": "ep-1.mkv", "kind": "video", "size_bytes": 1, "sha256": "c" * 64},
        ],
    }


def test_strict_manifest_accepts_only_v2_with_events():
    assert ShardManifestV2.model_validate(manifest()).action_spec_id == "switch_packets.v1"
    bad = manifest()
    bad["schema_id"] = "nxml.episode.v1"
    with pytest.raises(ValidationError):
        ShardManifestV2.model_validate(bad)
    bad = manifest()
    bad["members"] = bad["members"][:1]
    with pytest.raises(ValidationError, match=r"events.parquet"):
        ShardManifestV2.model_validate(bad)


def test_deterministic_human_and_policy_filtering():
    rows = [
        {"frame_index": 0, "valid": True, "human_action_mask": [False], "ownership": [0]},
        {"frame_index": 1, "valid": True, "human_action_mask": [True], "ownership": [0]},
        {"frame_index": 2, "valid": True, "human_action_mask": [False], "ownership": [1]},
        {"frame_index": 3, "valid": True, "human_action_mask": [False], "ownership": [2]},
        {"frame_index": 4, "valid": False, "human_action_mask": [True], "ownership": [1]},
    ]
    assert [r["frame_index"] for r in select_training_rows(rows, control_source="human")] == [1, 2]
    assert [r["frame_index"] for r in select_training_rows(rows, control_source="policy")] == [3]


def test_webdataset_materialization_is_member_sorted_and_human_filtered():
    import io
    import tarfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from nxml_control.materialize import iter_webdataset_rows

    def parquet(frame_index, owner):
        sink = io.BytesIO()
        pq.write_table(
            pa.table(
                {
                    "frame_index": frame_index,
                    "valid": [True] * len(frame_index),
                    "human_action_mask": [[False]] * len(frame_index),
                    "ownership": [[owner]] * len(frame_index),
                }
            ),
            sink,
        )
        return sink.getvalue()

    shard = io.BytesIO()
    with tarfile.open(fileobj=shard, mode="w") as archive:
        for name, data in (
            ("z.parquet", parquet([2], 1)),
            ("a.parquet", parquet([1], 1)),
            ("a.events.parquet", parquet([99], 1)),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    shard.seek(0)
    assert [row["frame_index"] for row in iter_webdataset_rows(shard, control_source="human")] == [
        1,
        2,
    ]
