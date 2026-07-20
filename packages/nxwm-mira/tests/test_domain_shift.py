from pathlib import Path

import pytest
import torch
from nxml_core.contracts.webdataset_v2 import CompactCodecLineageV2
from nxwm_mira.eval.domain_shift import (
    CodecCheckpointIdentity,
    CodecJobManifest,
    DomainShiftAuditInput,
    DomainShiftThresholds,
    EligibleEpisodeSelection,
    WindowMetrics,
    decide_domain_shift,
    deterministic_windows,
    measure_window,
    pad_to_frame_size,
    publication_dry_run,
    write_side_by_side,
)
from pydantic import ValidationError
from test_webdataset_episode import _manifest


def codec():
    return CompactCodecLineageV2(
        compatibility_id="nxml.compact-h264-720p60.v1",
        codec="h264",
        container="matroska",
        profile="High",
        level=42,
        pixel_format="yuv420p",
        width=1280,
        height=720,
        r_frame_rate="60/1",
        avg_frame_rate="60/1",
        time_base="1/1000",
        gop_size=60,
        max_b_frames=0,
        measured_bit_rate=16_000_000,
        aspect_mode="pad",
        artifact_probe="ffprobe",
    )


def selection(episode_id):
    return EligibleEpisodeSelection(
        episode_id=episode_id,
        split="train",
        disposition_id="quality-pass",
        validator="strict-gameplay",
        validator_version="1",
        training_eligible=True,
        human_owned_rows=100,
        dynamic_frame_fraction=0.5,
    )


def audit_input():
    return DomainShiftAuditInput(
        schema_id="nxml.codec-domain-shift-input.v1",
        source_snapshot_id="sha256:" + "a" * 64,
        hub_repo_id="arelius/nxml-pokemon-za-gameplay",
        hub_revision="b" * 40,
        compact_profile=codec(),
        checkpoint=CodecCheckpointIdentity(
            checkpoint_id="checkpoint-99999",
            checkpoint_digest="sha256:" + "c" * 64,
            config_digest="sha256:" + "d" * 64,
        ),
        legacy_episodes=[selection("legacy")],
        compact_episodes=[selection("compact")],
        seed=7,
        git_commit="e" * 40,
        audit_config_digest="sha256:" + "f" * 64,
        thresholds=DomainShiftThresholds(
            l1_relative_increase=0.2,
            lpips_relative_increase=0.2,
            dino_relative_increase=0.2,
            motion_relative_increase=0.2,
            action_conditioned_relative_increase=0.2,
            confidence_level=0.95,
            minimum_windows_per_domain=2,
        ),
    )


def test_padding_preserves_aspect_and_never_stretches():
    video = torch.ones(2, 3, 100, 100)
    padded = pad_to_frame_size(video, (288, 512))
    assert padded.shape == (2, 3, 288, 512)
    nonzero = torch.nonzero(padded[0, 0] > 0)
    assert tuple(nonzero.max(dim=0).values - nonzero.min(dim=0).values + 1) == (288, 288)


def test_window_sampling_is_seeded_deterministic_and_eligible_only():
    manifest = _manifest()
    selected = [selection("episode-0")]
    first = deterministic_windows(manifest, selected, seed=11, limit=2)
    second = deterministic_windows(manifest, selected, seed=11, limit=2)
    assert first == second
    assert {item.episode_id for item in first} == {"episode-0"}
    with pytest.raises(ValueError, match="not enough"):
        deterministic_windows(manifest, selected, seed=11, limit=99)


def test_bounded_metrics_and_side_by_side_artifact(tmp_path: Path):
    target = torch.zeros(4, 3, 8, 16)
    reconstruction = torch.ones_like(target)
    actions = torch.zeros(4, 26)
    metrics = measure_window(
        target,
        reconstruction,
        actions,
        feature_fn=lambda value: value.flatten(1) + 1,
        lpips_fn=lambda predicted, expected: (predicted - expected).abs().mean(),
    )
    assert metrics["l1"] == 1
    assert metrics["motion"] == 0
    artifact = tmp_path / "representative.png"
    digest = write_side_by_side(target, reconstruction, artifact)
    assert digest.startswith("sha256:") and artifact.stat().st_size > 0


def measured(domain, index, value):
    return WindowMetrics(
        sample_id="sha256:" + f"{index + (0 if domain == 'legacy' else 10):064x}",
        domain=domain,
        episode_id=domain,
        frame_indices=(0, 4, 8),
        l1=value,
        lpips=value,
        dino=value,
        motion=value,
        action_conditioned=value,
        artifact_sha256="sha256:" + f"{index + 30:064x}",
    )


def test_decision_is_auditable_and_requires_both_domains():
    audit = audit_input()
    results = [measured("legacy", 1, 1.0), measured("legacy", 2, 1.0)]
    with pytest.raises(ValueError, match="insufficient"):
        decide_domain_shift(audit, results)
    results += [measured("compact", 1, 2.0), measured("compact", 2, 2.0)]
    decision = decide_domain_shift(audit, results)
    assert decision.decision == "finetune_codec"
    assert decision.metrics["lpips"].exceeded is True
    assert decision.result_digest.startswith("sha256:")


def test_job_lineage_distinguishes_finetune_and_continue():
    common = {
        "schema_id": "nxml.codec-job.v1",
        "job_id": "bounded-codec-audit-result",
        "decision_digest": "sha256:" + "1" * 64,
        "source_snapshot_id": "sha256:" + "2" * 64,
        "codec_input_digest": "sha256:" + "3" * 64,
        "checkpoint_id": "checkpoint-99999",
        "max_steps": 30_000,
        "git_commit": "4" * 40,
        "config_digest": "sha256:" + "5" * 64,
    }
    assert (
        CodecJobManifest.model_validate({**common, "mode": "finetune_from"}).resume_job_id is None
    )
    with pytest.raises(ValidationError, match="exact prior job"):
        CodecJobManifest.model_validate({**common, "mode": "continue_from"})
    resumed = CodecJobManifest.model_validate(
        {**common, "mode": "continue_from", "resume_job_id": "same-job-before-crash"}
    )
    assert resumed.resume_job_id == "same-job-before-crash"


def test_publication_dry_run_refuses_veto_and_missing_inspection():
    eligible_segment = "sha256:" + "6" * 64
    snapshot = {
        "episodes": [{"episode_id": "good", "segment_ids": [eligible_segment]}],
        "excluded_episodes": [
            {
                "episode_id": "static",
                "episode_disposition": {"reason": "static_neutral_memory_bound_canary"},
            }
        ],
    }
    with pytest.raises(ValueError, match="excluded or vetoed"):
        publication_dry_run(snapshot, {"static"}, {})
    with pytest.raises(ValueError, match="missing or failed"):
        publication_dry_run(snapshot, {"good"}, {})
    result = publication_dry_run(
        snapshot,
        {"good"},
        {
            eligible_segment: {
                "decoded_frames_equal_action_rows": True,
                "codec_compatibility": codec().model_dump(mode="json"),
            }
        },
    )
    assert result.episode_ids == ("good",)
