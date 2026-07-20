from types import SimpleNamespace

import torch
from nxml_control.quality_inspect import inspect_actions, inspect_visual_dynamics


def row(index, action):
    frame_ns = 1_000_000_000 + index * 16_666_667
    return {
        "frame_index": index,
        "frame_timestamp_ns": frame_ns,
        "action_timestamp_ns": frame_ns - 1_000_000,
        "action_age_ns": 1_000_000,
        "applied_action": action,
        "human_action": action,
        "ownership": [1] * 26,
        "valid": True,
        "applied_action_valid": True,
        "bc_training_eligible": True,
    }


def test_action_quality_distinguishes_static_and_diverse_human_rows():
    neutral = [0.0] * 26
    active = [0.5, -0.5, 0.25, 0.0, *([0.0] * 20), 1.0, 0.0]
    quality = inspect_actions([row(0, neutral), row(1, active), row(2, active)])
    assert quality["human_owned_rows"] == 3
    assert quality["bc_training_eligible_rows"] == 3
    assert quality["non_neutral_rows"] == 2
    assert quality["distinct_applied_action_count"] == 2
    assert quality["control_activation_counts"]["L_STICK_X"] == 2
    assert quality["control_activation_counts"]["B"] == 2
    assert quality["action_transitions"] == 1
    assert quality["run_length_frames"]["max"] == 2


class Decoder:
    def __init__(self, dynamic):
        self.dynamic = dynamic

    def get_frames_at(self, indices):
        frames = torch.zeros(len(indices), 3, 16, 16)
        if self.dynamic:
            frames[1::2] = 255
        return SimpleNamespace(data=frames)


def test_visual_quality_separates_static_and_dynamic_samples():
    static = inspect_visual_dynamics(Decoder(False), 120, "sha256:" + "1" * 64)
    dynamic = inspect_visual_dynamics(Decoder(True), 120, "sha256:" + "1" * 64)
    assert static["repeated_frame_fraction"] == 1
    assert static["motion_active_fraction"] == 0
    assert dynamic["repeated_frame_fraction"] == 0
    assert dynamic["motion_active_fraction"] == 1
    assert dynamic["sampled_frame_count"] <= 32
    assert dynamic["sampling_stride_frames"] == 4
