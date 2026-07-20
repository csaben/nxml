from nxml_control.hf_webdataset import stage_segment_snapshot

from tests.nxml_control.test_bc_worker import committed_segment_snapshot


def codec_lineage():
    return {
        "codec": "h264",
        "container": "matroska",
        "profile": "High",
        "level": 42,
        "pixel_format": "yuv420p",
        "width": 1280,
        "height": 720,
        "nominal_fps": 60.0,
        "time_base": "1/1000",
        "gop_size": 60,
        "aspect_mode": "pad",
    }


def test_stage_snapshot_is_bounded_restart_safe_and_never_deletes_cluster_objects(tmp_path):
    _client, snapshot, manifests = committed_segment_snapshot(tmp_path)
    episode_id = snapshot["episodes"][0]["episode_id"]
    output = tmp_path / "publication"
    publication, manifest_path = stage_segment_snapshot(
        state_dir=tmp_path,
        snapshot_id=snapshot["snapshot_id"],
        output_dir=output,
        codec_by_episode={episode_id: codec_lineage()},
        split_by_episode={episode_id: "train"},
        max_shards=1,
    )
    assert len(publication.shards) == 1
    assert publication.shards[0].segment_id == snapshot["episodes"][0]["segment_ids"][0]
    assert manifest_path.is_file()
    object_files = list((tmp_path / "objects").rglob("*.tar"))
    assert len(object_files) == len(manifests)
    before = [(path, path.stat().st_size) for path in object_files]
    repeated, _ = stage_segment_snapshot(
        state_dir=tmp_path,
        snapshot_id=snapshot["snapshot_id"],
        output_dir=output,
        codec_by_episode={episode_id: codec_lineage()},
        split_by_episode={episode_id: "train"},
        max_shards=1,
    )
    assert repeated == publication
    assert [(path, path.stat().st_size) for path in object_files] == before


def test_stage_snapshot_fails_closed_without_codec_lineage(tmp_path):
    _client, snapshot, _manifests = committed_segment_snapshot(tmp_path)
    try:
        stage_segment_snapshot(
            state_dir=tmp_path,
            snapshot_id=snapshot["snapshot_id"],
            output_dir=tmp_path / "publication",
            codec_by_episode={},
            split_by_episode={},
            max_shards=1,
        )
    except ValueError as error:
        assert "missing immutable codec/split lineage" in str(error)
    else:
        raise AssertionError("missing codec lineage must fail closed")
