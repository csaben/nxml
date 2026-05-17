"""Latent BC dataset.

Each ``.npz`` carries ``{latents: (T, C, H, W) float16, actions: (T, 26) float32}``
(produced by ``nxwm encode``). Indexes contiguous windows of length
``sequence_length``; the action target is the action at the last frame of
the window. With ``align_starts=True`` we instead chain consecutive
episode parts until we have ``>= sequence_length`` frames — useful for
"always start at t=0" finetune workloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentBCDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        *,
        sequence_length: int = 10,
        align_starts: bool = False,
    ) -> None:
        self.files = list(files)
        self.sequence_length = sequence_length
        self.align_starts = align_starts
        self.indices: list[tuple[int, int]] = []
        self.episode_groups: list[list[int]] = []
        self.mapped_data: dict[int, np.lib.npyio.NpzFile] = {}

        frame_counts: list[int] = []
        for f_idx, f_path in enumerate(self.files):
            data = np.load(f_path, mmap_mode="r")
            self.mapped_data[f_idx] = data
            frame_counts.append(data["latents"].shape[0])

        if align_starts:
            group: list[int] = []
            group_frames = 0
            for f_idx, n in enumerate(frame_counts):
                group.append(f_idx)
                group_frames += n
                if group_frames >= sequence_length:
                    self.episode_groups.append(group)
                    group = []
                    group_frames = 0
        else:
            for f_idx, n in enumerate(frame_counts):
                for start_frame in range(n - sequence_length):
                    self.indices.append((f_idx, start_frame))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["mapped_data"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for f_idx, f_path in enumerate(self.files):
            self.mapped_data[f_idx] = np.load(f_path, mmap_mode="r")

    def __len__(self) -> int:
        if self.align_starts:
            return len(self.episode_groups)
        return len(self.indices)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:  # type: ignore[override]
        if self.align_starts:
            group = self.episode_groups[idx]
            latent_chunks = [self.mapped_data[f]["latents"][:] for f in group]
            action_chunks = [self.mapped_data[f]["actions"][:] for f in group]
            all_latents = np.concatenate(latent_chunks, axis=0)
            all_actions = np.concatenate(action_chunks, axis=0)
            obs = torch.from_numpy(all_latents[: self.sequence_length].copy())
            action = torch.from_numpy(all_actions[self.sequence_length - 1].copy())
            return {"observations": obs.float(), "action": action.float()}

        f_idx, start = self.indices[idx]
        data = self.mapped_data[f_idx]
        obs = torch.from_numpy(data["latents"][start : start + self.sequence_length].copy())
        action = torch.from_numpy(data["actions"][start + self.sequence_length - 1].copy())
        return {"observations": obs.float(), "action": action.float()}
