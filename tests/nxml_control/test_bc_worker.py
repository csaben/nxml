import hashlib
import io
import tarfile

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
                "proposal_valid": False,
                "proposal_fresh": False,
                "bc_training_eligible": eligible,
                "valid": True,
                "invalid_reasons": [],
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()


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
