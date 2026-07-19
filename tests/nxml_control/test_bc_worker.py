import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from nxml_control.api import create_app
from nxml_control.bc_worker import (
    ACTION_DIM,
    VAE_PROFILE,
    BootstrapConfig,
    _split_episode,
    _training_config,
    decode_action_rows,
    prepare_snapshot,
)


def action_parquet(*, count=80, noncausal_at=None):
    sink = io.BytesIO()
    actions = [[float(index % 3) / 2, 0.0, 0.0, 0.0, *([0.0] * 22)] for index in range(count)]
    frame_ns = [1_000_000_000 + index * 33_333_333 for index in range(count)]
    action_ns = [value - 1_000_000 for value in frame_ns]
    if noncausal_at is not None:
        action_ns[noncausal_at] = frame_ns[noncausal_at] + 1
    pq.write_table(
        pa.table(
            {
                "frame_idx": list(range(count)),
                "frame_monotonic_ns": frame_ns,
                "action_monotonic_ns": action_ns,
                "action_age": [
                    max(frame - action, 0) / 1_000_000_000
                    for frame, action in zip(frame_ns, action_ns, strict=True)
                ],
                "applied_action": actions,
                "human_mask": [[True, *([False] * 25)] for _ in range(count)],
                "ownership": [[1, *([0] * 25)] for _ in range(count)],
                "valid": [True] * count,
            }
        ),
        sink,
    )
    return sink.getvalue()


def dagger_action_parquet(*, count=80, eligible=True):
    sink = io.BytesIO()
    rows = []
    for index in range(count):
        frame_ns = 1_000_000_000 + index * 33_333_333
        action_ns = frame_ns - 1_000_000
        human = [float(index % 2), *([0.0] * 25)]
        rows.append(
            {
                "frame_idx": index,
                "timestamp": frame_ns / 1_000_000_000,
                "frame_monotonic_ns": frame_ns,
                "frame_timestamp_ns": frame_ns,
                "action_timestamp": action_ns / 1_000_000_000,
                "action_monotonic_ns": action_ns,
                "action_timestamp_ns": action_ns,
                "action_age_ns": 1_000_000,
                "action_age": 0.001,
                "action": human,
                "applied_action": human,
                "human_action": human,
                "human_mask": [True, *([False] * 25)],
                "policy_action": [0.0] * 26,
                "muted_policy_action": [0.0] * 26,
                "mute_mask": [False] * 26,
                "mute_mask_version": "switch_packets.v1/mute.v1",
                "ownership": [1] * 26,
                "controller_id": "dagger-arbitrator:switch_packets.v1",
                "active_driver": "human",
                "controller": "human",
                "policy_id": None,
                "policy_revision": None,
                "policy_digest": None,
                "human_monotonic_ns": action_ns,
                "policy_monotonic_ns": None,
                "policy_observation_monotonic_ns": None,
                "ownership_source": "human",
                "mode": "human",
                "takeover": False,
                "takeover_reason": None,
                "takeover_release_remaining_ns": 0,
                "proposal_valid": False,
                "proposal_fresh": False,
                "proposal_sequence": None,
                "proposal_age_ns": None,
                "gap_state": "none",
                "gap_reason": None,
                "gap_duration_ns": 0,
                "boundary_sequence": None,
                "boundary_acknowledged": False,
                "bc_training_eligible": eligible,
                "valid": True,
                "invalid_reasons": [],
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()



def rolling_segment_tar(episode_id, index, start_ns, end_ns, *, eligible=True):
    members = {
        f"{episode_id}.{index:06d}.mkv": b"test-video-not-decoded",
        f"{episode_id}.{index:06d}.parquet": dagger_action_parquet(count=40, eligible=eligible),
        f"{episode_id}.{index:06d}.events.parquet": action_parquet(count=1),
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name); info.size = len(content); archive.addfile(info, io.BytesIO(content))
    content = stream.getvalue(); digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_id": "nxml.segment-bundle.v1", "dataset_id": "rolling",
        "episode_id": episode_id, "segment_id": "sha256:" + digest,
        "sequence_index": index, "clock_id": "linux-monotonic",
        "timeline_start_ns": start_ns, "timeline_end_ns": end_ns,
        "object_size_bytes": len(content), "object_sha256": digest,
        "members": [{"role": "events" if name.endswith(".events.parquet") else ("actions" if name.endswith(".parquet") else "video"), "path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members.items()],
    }
    return content, manifest


def committed_segment_snapshot(tmp_path, *, eligible=True):
    episode_id = "11111111-1111-4111-8111-111111111111"
    client = TestClient(create_app(state_dir=tmp_path))
    manifests = []
    for index in range(2):
        content, manifest = rolling_segment_tar(episode_id, index, index * 100, (index + 1) * 100, eligible=eligible)
        upload = client.post("/v1/uploads", headers={"Idempotency-Key": f"rolling-{index}"}, json={"object_key": f"uploads/segments/{episode_id}/{index:06d}-{manifest['object_sha256']}.tar", "size_bytes": len(content), "sha256": manifest["object_sha256"]}).json()
        assert client.put(upload["upload_url"], content=content).status_code == 200
        assert client.post(f"/v1/segment-bundles/{upload['id']}/commit", json=manifest).status_code == 200
        assert client.post(f"/v1/segments/{manifest['segment_id']}/quality-dispositions", headers={"Idempotency-Key": f"quality-{index}"}, json={"training_eligible": True, "reason": "fixture_passed", "validator": "fixture", "validator_version": "1", "validator_state": "passed"}).status_code == 201
        manifests.append(manifest)
    close = {"schema_id": "nxml.episode-close.v1", "dataset_id": "rolling", "episode_id": episode_id, "clock_id": "linux-monotonic", "timeline_start_ns": 0, "timeline_end_ns": 200, "segments": [{"segment_id": item["segment_id"], "sequence_index": item["sequence_index"], "timeline_start_ns": item["timeline_start_ns"], "timeline_end_ns": item["timeline_end_ns"], "object_sha256": item["object_sha256"]} for item in manifests]}
    assert client.post(f"/v1/datasets/rolling/episodes/{episode_id}/close", headers={"Idempotency-Key": "rolling-close"}, json=close).status_code == 201
    assert client.post(f"/v1/datasets/rolling/episodes/{episode_id}/quality-dispositions", headers={"Idempotency-Key": "episode-quality"}, json={"training_eligible": True, "reason": "fixture_passed", "validator": "fixture", "validator_version": "1"}).status_code == 201
    snapshot = client.post("/v1/datasets/rolling/segment-snapshots").json()
    return client, snapshot, manifests


def test_actual_edge_physical_parquet_round_trips_strictly() -> None:
    from nxml_core.contracts import DaggerActionRecordV2

    rows = pq.read_table(io.BytesIO(dagger_action_parquet(count=3))).to_pylist()
    assert len(rows) == 3
    for row in rows:
        parsed = DaggerActionRecordV2.model_validate(row)
        assert parsed.model_dump(mode="json", exclude_unset=True) == row
        assert parsed.bc_training_eligible is True
        assert parsed.gap_state.value == "none"
        assert parsed.boundary_sequence is None


def raw_shard(episode_id="human", *, noncausal_at=None):
    members = {
        f"{episode_id}.mkv": b"test-video-not-decoded",
        f"{episode_id}.parquet": action_parquet(noncausal_at=noncausal_at),
        f"{episode_id}.events.parquet": action_parquet(count=1),
    }
    kinds = {".mkv": "video", ".events.parquet": "events", ".parquet": "actions"}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    manifest = {
        "schema_id": "nxml.episode.v2",
        "action_spec_id": "switch_packets.v1",
        "episodes": [{"episode_id": episode_id}],
        "members": [
            {
                "path": name,
                "kind": next(value for suffix, value in kinds.items() if name.endswith(suffix)),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in members.items()
        ],
    }
    return stream.getvalue(), manifest


def committed_human_snapshot(tmp_path):
    client = TestClient(create_app(state_dir=tmp_path))
    content, manifest = raw_shard()
    digest = hashlib.sha256(content).hexdigest()
    upload = client.post(
        "/v1/uploads",
        headers={"Idempotency-Key": "worker-shard"},
        json={"object_key": "uploads/worker.tar", "size_bytes": len(content), "sha256": digest},
    ).json()
    assert client.put(upload["upload_url"], content=content).status_code == 200
    assert (
        client.post(
            f"/v1/uploads/{upload['id']}/commit",
            json={"dataset_id": "raw", "shard_id": "worker-shard", "manifest": manifest},
        ).status_code
        == 200
    )
    return client.post("/v1/datasets/raw/snapshots", json={"control_source": "human"}).json()


def fake_encoder(_video, indices, **_kwargs):
    return np.stack([np.full((4, 16, 32), index, dtype=np.float16) for index in indices])



def test_prepare_segment_snapshot_reconstructs_ordered_trainable_episode(tmp_path):
    client, snapshot, manifests = committed_segment_snapshot(tmp_path)
    prepared = prepare_snapshot(state_dir=tmp_path, snapshot_id=snapshot["snapshot_id"], workspace=tmp_path / "rolling-worker", config=BootstrapConfig(sequence_length=8, validation_fraction=0.2), encode_fn=fake_encoder, device="cpu")
    assert prepared.episodes == ["11111111-1111-4111-8111-111111111111"]
    assert (prepared.train_frames, prepared.val_frames) == (64, 16)
    assert [item["segment_id"] for item in prepared.source_members[::2]] == snapshot["episodes"][0]["segment_ids"]
    submitted = client.post("/v1/training/jobs", headers={"Idempotency-Key": "rolling-job-lookup"}, json={"snapshot_id": snapshot["snapshot_id"], "config": {}})
    assert submitted.status_code == 201


def test_prepare_segment_snapshot_rejects_missing_or_corrupt_object(tmp_path):
    _client, snapshot, manifests = committed_segment_snapshot(tmp_path)
    catalog = _client.app.state.catalog
    with catalog.connect() as db:
        row = db.execute("SELECT storage_key FROM segment_bundles WHERE segment_id=?", (manifests[0]["segment_id"],)).fetchone()
    object_path = tmp_path / "objects" / row["storage_key"]
    original = object_path.read_bytes(); object_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cluster segment verification failed"):
        prepare_snapshot(state_dir=tmp_path, snapshot_id=snapshot["snapshot_id"], workspace=tmp_path / "corrupt-worker", config=BootstrapConfig(sequence_length=8), encode_fn=fake_encoder, device="cpu")
    object_path.write_bytes(original); object_path.unlink()
    with pytest.raises(ValueError, match="cluster segment verification failed"):
        prepare_snapshot(state_dir=tmp_path, snapshot_id=snapshot["snapshot_id"], workspace=tmp_path / "missing-worker", config=BootstrapConfig(sequence_length=8), encode_fn=fake_encoder, device="cpu")


def test_prepare_segment_snapshot_rejects_order_drift_and_exclusions(tmp_path):
    client, snapshot, _manifests = committed_segment_snapshot(tmp_path)
    with client.app.state.catalog.connect() as db:
        body = json.loads(db.execute("SELECT manifest_json FROM segment_snapshots WHERE snapshot_id=?", (snapshot["snapshot_id"],)).fetchone()[0])
        body["episodes"][0]["segment_ids"].reverse()
        db.execute("UPDATE segment_snapshots SET manifest_json=? WHERE snapshot_id=?", (json.dumps(body), snapshot["snapshot_id"]))
    with pytest.raises(ValueError, match="ordering/identity mismatch"):
        prepare_snapshot(state_dir=tmp_path, snapshot_id=snapshot["snapshot_id"], workspace=tmp_path / "order-worker", config=BootstrapConfig(sequence_length=8), encode_fn=fake_encoder, device="cpu")

    other = tmp_path / "excluded"; client, _snapshot, _ = committed_segment_snapshot(other)
    episode_id = "11111111-1111-4111-8111-111111111111"
    assert client.post(f"/v1/datasets/rolling/episodes/{episode_id}/quality-dispositions", headers={"Idempotency-Key": "episode-veto"}, json={"training_eligible": False, "reason": "fixture_veto", "validator": "fixture", "validator_version": "1"}).status_code == 201
    excluded = client.post("/v1/datasets/rolling/segment-snapshots").json()
    assert excluded["episodes"] == [] and excluded["excluded_episodes"][0]["episode_id"] == episode_id
    with pytest.raises(ValueError, match="no materializable eligible episodes"):
        prepare_snapshot(state_dir=other, snapshot_id=excluded["snapshot_id"], workspace=other / "worker", config=BootstrapConfig(sequence_length=8), encode_fn=fake_encoder, device="cpu")


def test_prepare_snapshot_verifies_and_materializes_edge_compatible_data(tmp_path):
    snapshot = committed_human_snapshot(tmp_path)
    config = BootstrapConfig(sequence_length=8, validation_fraction=0.2)
    prepared = prepare_snapshot(
        state_dir=tmp_path,
        snapshot_id=snapshot["snapshot_id"],
        workspace=tmp_path / "worker",
        config=config,
        encode_fn=fake_encoder,
        device="cpu",
    )
    assert prepared.episodes == ["human"]
    assert prepared.train_frames == 64 and prepared.val_frames == 16
    train = np.load(prepared.train_files[0])
    val = np.load(prepared.val_files[0])
    assert train["latents"].shape == (64, 4, 16, 32)
    assert train["actions"].shape == (64, ACTION_DIM)
    assert float(train["latents"][-1, 0, 0, 0]) == 63
    assert float(val["latents"][0, 0, 0, 0]) == 64
    generated = _training_config(prepared, config, tmp_path / "checkpoints")
    assert generated["policy"]["name"] == "bc_transformer_v1"
    assert generated["policy"]["config"]["sequence_length"] == 8
    assert generated["data"]["val_files"] == prepared.val_files
    assert VAE_PROFILE.endswith("scale-0.18215.v1")


def test_action_decoder_rejects_hidden_negative_causal_age(tmp_path):
    parquet = tmp_path / "bad.parquet"
    parquet.write_bytes(action_parquet(noncausal_at=3))
    with pytest.raises(ValueError, match="noncausal action alignment"):
        decode_action_rows(parquet, control_source="human")


def test_action_decoder_accepts_dagger_rows_and_uses_applied_action_only(tmp_path):
    parquet = tmp_path / "dagger.parquet"
    parquet.write_bytes(dagger_action_parquet(count=8))
    indices, actions, total = decode_action_rows(parquet, control_source="human")
    assert indices == list(range(8)) and total == 8
    assert actions.shape == (8, 26)
    assert actions[:, 0].tolist() == [float(index % 2) for index in range(8)]

    parquet.write_bytes(dagger_action_parquet(count=8, eligible=False))
    with pytest.raises(ValueError, match="no eligible action rows"):
        decode_action_rows(parquet, control_source="human")

    deployed = pq.read_table(io.BytesIO(dagger_action_parquet(count=8))).drop(
        ["bc_training_eligible"]
    )
    pq.write_table(deployed, parquet)
    with pytest.raises(ValueError, match="no eligible action rows"):
        decode_action_rows(parquet, control_source="human")


def test_action_decoder_excludes_valid_neutral_boundary_rows(tmp_path):
    rows = pq.read_table(io.BytesIO(dagger_action_parquet(count=8))).to_pylist()
    zero = [0.0] * 26
    rows[0].update(
        {
            "action": zero,
            "applied_action": zero,
            "ownership": [0] * 26,
            "bc_training_eligible": False,
            "boundary_sequence": 1,
            "boundary_acknowledged": True,
        }
    )
    parquet = tmp_path / "boundary.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    indices, _actions, total = decode_action_rows(parquet, control_source="human")
    assert indices == list(range(1, 8))
    assert total == 8


def test_action_decoder_accepts_invalid_neutral_dagger_row_with_null_action_time(tmp_path):
    rows = pq.read_table(io.BytesIO(dagger_action_parquet(count=8))).to_pylist()
    zero = [0.0] * 26
    rows[0].update(
        {
            "action_timestamp": None,
            "action_monotonic_ns": None,
            "action_timestamp_ns": None,
            "action_age_ns": None,
            "action_age": 0.0,
            "action": zero,
            "applied_action": zero,
            "human_action": zero,
            "human_mask": [False] * 26,
            "policy_action": zero,
            "muted_policy_action": zero,
            "ownership": [0] * 26,
            "controller_id": None,
            "active_driver": "none",
            "controller": "none",
            "human_monotonic_ns": None,
            "ownership_source": None,
            "mode": None,
            "bc_training_eligible": False,
            "valid": False,
            "invalid_reasons": ["no_prior_arbitration_record"],
        }
    )
    parquet = tmp_path / "invalid-neutral.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    indices, _, total = decode_action_rows(parquet, control_source="human")
    assert total == 8
    assert indices == list(range(1, 8))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.drop(["frame_idx"]), "lacks required fields"),
        (
            lambda table: table.set_column(
                table.schema.get_field_index("frame_idx"),
                "frame_idx",
                pa.array([0, 0, *range(2, table.num_rows)]),
            ),
            "complete, ordered, duplicate-free",
        ),
        (
            lambda table: table.set_column(
                table.schema.get_field_index("frame_idx"),
                "frame_idx",
                pa.array([1, 0, *range(2, table.num_rows)]),
            ),
            "complete, ordered, duplicate-free",
        ),
    ],
)
def test_action_decoder_rejects_missing_duplicate_or_out_of_order_sequence(
    tmp_path, mutation, message
):
    source = pq.read_table(io.BytesIO(action_parquet(count=8)))
    parquet = tmp_path / "invalid.parquet"
    pq.write_table(mutation(source), parquet)
    with pytest.raises(ValueError, match=message):
        decode_action_rows(parquet, control_source="human")


def test_prepare_snapshot_requires_one_media_frame_per_action_row(tmp_path):
    snapshot = committed_human_snapshot(tmp_path)

    def mismatched_encoder(_video, _indices, *, expected_frame_count, **_kwargs):
        assert expected_frame_count == 80
        raise ValueError(
            "media frame count does not match the complete edge-v2 action row sequence"
        )

    with pytest.raises(ValueError, match="media frame count does not match"):
        prepare_snapshot(
            state_dir=tmp_path,
            snapshot_id=snapshot["snapshot_id"],
            workspace=tmp_path / "mismatch-worker",
            config=BootstrapConfig(sequence_length=8, validation_fraction=0.2),
            encode_fn=mismatched_encoder,
            device="cpu",
        )


def test_split_requires_independent_sequence_windows():
    latents = np.zeros((17, 4, 16, 32), dtype=np.float16)
    actions = np.zeros((17, ACTION_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 18"):
        _split_episode(latents, actions, sequence_length=8, val_fraction=0.1)


def _rolling_validator_module():
    path = Path(__file__).parents[2] / "apps" / "nxml-control" / "deploy" / "validate_rolling.py"
    spec = importlib.util.spec_from_file_location("nxml_validate_rolling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rolling_validator_accepts_canonical_edge_shape_and_schema_aliases() -> None:
    helper = _rolling_validator_module()
    columns = sorted(helper.EDGE_ACTION_COLUMNS)
    assert helper.validate_action_schema(columns, [{}]) == helper.ACTION_SCHEMA_ID
    for marker in helper.ACTION_SCHEMA_MARKERS:
        assert (
            helper.validate_action_schema(
                [*columns, marker], [{marker: helper.ACTION_SCHEMA_ID}]
            )
            == helper.ACTION_SCHEMA_ID
        )


def test_rolling_validator_rejects_schema_shape_and_marker_drift() -> None:
    helper = _rolling_validator_module()
    columns = sorted(helper.EDGE_ACTION_COLUMNS)
    with pytest.raises(ValueError, match="missing=.*proposal_sequence"):
        helper.validate_action_schema(
            [name for name in columns if name != "proposal_sequence"], [{}]
        )
    with pytest.raises(ValueError, match="extra=.*mystery"):
        helper.validate_action_schema([*columns, "mystery"], [{}])
    with pytest.raises(ValueError, match="encountered=.*nxml.dagger-actions.v1"):
        helper.validate_action_schema(
            [*columns, "action_schema_id"],
            [{"action_schema_id": "nxml.dagger-actions.v1"}],
        )
