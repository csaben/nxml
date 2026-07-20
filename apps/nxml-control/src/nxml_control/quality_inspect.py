"""Bounded action and visual quality metrics for immutable segment inspection."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np
import torch.nn.functional as F
from nxml_core.contracts import DaggerActionRecordV2

BUTTON_NAMES = (
    "L_STICK_PRESSED",
    "R_STICK_PRESSED",
    "DPAD_UP",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "DPAD_DOWN",
    "L",
    "ZL",
    "R",
    "ZR",
    "JCL_SR",
    "JCL_SL",
    "JCR_SR",
    "JCR_SL",
    "PLUS",
    "MINUS",
    "HOME",
    "CAPTURE",
    "Y",
    "X",
    "B",
    "A",
)
CONTROL_NAMES = ("L_STICK_X", "L_STICK_Y", "R_STICK_X", "R_STICK_Y", *BUTTON_NAMES)
MAX_VISUAL_SAMPLES = 32
MAX_DISTINCT_HASHES = 2048


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.percentile(values, value))
        for name, value in (("p50", 50), ("p90", 90), ("p99", 99))
    }


def inspect_actions(rows: list[dict]) -> dict:
    parsed = [DaggerActionRecordV2.model_validate(row) for row in rows]
    actions = np.asarray([row.applied_action for row in parsed], dtype=np.float32)
    if actions.shape != (len(rows), 26):
        raise ValueError("action matrix must be Nx26")
    active = np.concatenate((np.abs(actions[:, :4]) >= 0.1, actions[:, 4:] >= 0.5), axis=1)
    packet_hashes = [hashlib.sha256(action.tobytes()).hexdigest() for action in actions]
    distinct = sorted(set(packet_hashes))
    runs = []
    start = 0
    for index in range(1, len(packet_hashes) + 1):
        if index == len(packet_hashes) or packet_hashes[index] != packet_hashes[start]:
            runs.append((packet_hashes[start], index - start))
            start = index
    ownership_dimensions = Counter(int(owner) for row in parsed for owner in row.ownership)
    human_rows = sum(any(int(owner) == 1 for owner in row.ownership) for row in parsed)
    policy_rows = sum(any(int(owner) == 2 for owner in row.ownership) for row in parsed)
    eligible_rows = sum(row.bc_training_eligible for row in parsed)
    valid_rows = sum(row.valid and row.applied_action_valid is not False for row in parsed)
    left = np.linalg.norm(actions[:, :2], axis=1)
    right = np.linalg.norm(actions[:, 2:4], axis=1)
    run_lengths = np.asarray([length for _packet, length in runs], dtype=np.float64)
    return {
        "row_count": len(parsed),
        "valid_rows": valid_rows,
        "invalid_rows": len(parsed) - valid_rows,
        "bc_training_eligible_rows": eligible_rows,
        "human_owned_rows": human_rows,
        "human_owned_fraction": human_rows / len(parsed),
        "policy_owned_rows": policy_rows,
        "ownership_dimension_counts": {
            "unowned": ownership_dimensions[0],
            "human": ownership_dimensions[1],
            "policy": ownership_dimensions[2],
        },
        "non_neutral_rows": int(active.any(axis=1).sum()),
        "non_neutral_fraction": float(active.any(axis=1).mean()),
        "control_activation_counts": {
            name: int(active[:, index].sum()) for index, name in enumerate(CONTROL_NAMES)
        },
        "left_stick_magnitude": _percentiles(left),
        "right_stick_magnitude": _percentiles(right),
        "button_active_rows": int(active[:, 4:].any(axis=1).sum()),
        "trigger_active_rows": int(active[:, [11, 13]].any(axis=1).sum()),
        "distinct_applied_action_count": len(distinct),
        "distinct_applied_action_set_sha256": hashlib.sha256(
            "".join(distinct).encode()
        ).hexdigest(),
        "distinct_applied_action_hashes": distinct[:MAX_DISTINCT_HASHES],
        "distinct_hashes_truncated": len(distinct) > MAX_DISTINCT_HASHES,
        "action_transitions": max(0, len(runs) - 1),
        "sustained_runs": len(runs),
        "distinct_run_actions": len({packet for packet, _length in runs}),
        "run_length_frames": {
            **_percentiles(run_lengths),
            "max": int(run_lengths.max()),
        },
    }


def inspect_visual_dynamics(decoder, frame_count: int, segment_id: str) -> dict:
    if frame_count < 2:
        raise ValueError("visual dynamics require at least two frames")
    seed = int(segment_id.removeprefix("sha256:")[:16], 16)
    stride = max(1, math.ceil(frame_count / MAX_VISUAL_SAMPLES))
    offset = seed % stride
    indices = list(range(offset, frame_count, stride))[:MAX_VISUAL_SAMPLES]
    if len(indices) < 2:
        indices = [0, frame_count - 1]
    frames = decoder.get_frames_at(indices).data.float()
    small = F.interpolate(frames, size=(90, 160), mode="bilinear", antialias=True) / 255.0
    differences = (small[1:] - small[:-1]).abs()
    rgb = differences.flatten(1).mean(dim=1).cpu().numpy()
    luma_frames = small[:, 0] * 0.2126 + small[:, 1] * 0.7152 + small[:, 2] * 0.0722
    luma = (luma_frames[1:] - luma_frames[:-1]).abs().flatten(1).mean(dim=1).cpu().numpy()
    repeated_threshold = 1 / 255
    motion_threshold = 2 / 255
    sample_hash = hashlib.sha256()
    sample_hash.update(np.asarray(indices, dtype=np.int64).tobytes())
    sample_hash.update(small.mul(255).byte().cpu().numpy().tobytes())
    return {
        "sampling_seed": seed,
        "sampling_stride_frames": stride,
        "sampled_frame_count": len(indices),
        "sampled_pair_count": len(indices) - 1,
        "sampled_frame_indices_sha256": hashlib.sha256(
            np.asarray(indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "sampled_pixels_sha256": sample_hash.hexdigest(),
        "downsample_hw": [90, 160],
        "rgb_frame_difference": _percentiles(rgb),
        "luma_frame_difference": _percentiles(luma),
        "repeated_frame_threshold": repeated_threshold,
        "repeated_frame_fraction": float((luma <= repeated_threshold).mean()),
        "motion_active_threshold": motion_threshold,
        "motion_active_fraction": float((luma > motion_threshold).mean()),
    }
