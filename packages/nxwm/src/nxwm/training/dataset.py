"""Dataset for pre-baked .npz episode files (mmap-backed).

Each .npz contains:
    {"latents": (T, C, H, W) float16, "actions": (T, A) float32}

Produces a dict per index (keys after ``future_actions`` only emitted when
the corresponding constructor knob is non-zero, i.e. for fa_dit):
    {
      "observations":   (sequence_length, C, H, W) float32,
      "actions":        (sequence_length, A) float32,
      "targets":        (unroll_steps, C, H, W) float32,
      "goals":          (unroll_steps, C, H, W) float32,
      "future_actions": (unroll_steps, A) float32,
      "future_latents": (future_anchor_len, C, H, W) float32,  # if >0
      "future_poses":   (future_anchor_len, A) float32,        # if >0
      "plan_target":    (plan_horizon, A) float32,             # if >0
    }

Latents are emitted unscaled — the trainer applies ``LATENT_SCALE`` if
needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentEpisodeDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        *,
        sequence_length: int = 10,
        unroll_steps: int = 1,
        goal_offset: int = 30,
        future_anchor_len: int = 0,
        plan_horizon: int = 0,
    ):
        self.files = list(files)
        self.sequence_length = sequence_length
        self.unroll_steps = unroll_steps
        self.goal_offset = goal_offset
        self.future_anchor_len = future_anchor_len
        self.plan_horizon = plan_horizon
        self.indices: list[tuple[int, int]] = []
        self.mapped_data: dict[int, np.lib.npyio.NpzFile] = {}

        # Need room for: history + K unroll targets + the deepest lookahead.
        # fa_dit's future anchor and plan target both start from the target
        # window (t0); the latest frame any consumer reads is max of:
        #   - goal_offset + K            (dit_v1 goal at last unroll step)
        #   - goal_offset + future_anchor_len  (fa_dit anchor window end)
        #   - plan_horizon               (fa_dit plan target end, from t0)
        lookahead = max(
            goal_offset + unroll_steps,
            goal_offset + future_anchor_len,
            plan_horizon,
        )
        min_frames = sequence_length + lookahead

        for f_idx, f_path in enumerate(self.files):
            data = np.load(f_path, mmap_mode="r")
            self.mapped_data[f_idx] = data
            num_frames = data["latents"].shape[0]
            for start_frame in range(num_frames - min_frames):
                self.indices.append((f_idx, start_frame))

    def __getstate__(self):
        # NpzFile handles aren't picklable; drop them and re-open in __setstate__.
        state = self.__dict__.copy()
        state["mapped_data"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for f_idx, f_path in enumerate(self.files):
            self.mapped_data[f_idx] = np.load(f_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:  # type: ignore[override]
        f_idx, start = self.indices[idx]
        data = self.mapped_data[f_idx]
        K = self.unroll_steps
        T = self.sequence_length
        t0 = start + T

        obs = torch.from_numpy(data["latents"][start : start + T])
        acts = torch.from_numpy(data["actions"][start : start + T])
        targets = torch.from_numpy(data["latents"][t0 : t0 + K])
        goals = torch.from_numpy(
            data["latents"][t0 + self.goal_offset : t0 + K + self.goal_offset]
        )
        future_actions = torch.from_numpy(data["actions"][t0 : t0 + K])

        out: dict[str, torch.Tensor] = {
            "observations": obs.float(),
            "actions": acts.float(),
            "targets": targets.float(),
            "goals": goals.float(),
            "future_actions": future_actions.float(),
        }

        # fa_dit extras: anchor window F_a frames ahead, plan-head target.
        if self.future_anchor_len > 0:
            anchor_start = t0 + self.goal_offset
            F_a = self.future_anchor_len
            out["future_latents"] = torch.from_numpy(
                data["latents"][anchor_start : anchor_start + F_a]
            ).float()
            out["future_poses"] = torch.from_numpy(
                data["actions"][anchor_start : anchor_start + F_a]
            ).float()
        if self.plan_horizon > 0:
            out["plan_target"] = torch.from_numpy(
                data["actions"][t0 : t0 + self.plan_horizon]
            ).float()
        return out
