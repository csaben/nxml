"""Protocols for nxrl policies and algorithms.

A ``Policy`` is the runtime artifact: an ``nn.Module`` whose ``forward``
takes a sequence of latents and returns a 26-dim action vector
(``sticks[:4]`` tanh'd in [-1, 1] and ``buttons[4:]`` raw logits — apply
sigmoid + threshold for inference). A policy also exposes its
``sequence_length`` so callers know the input window size.

An ``Algorithm`` ties (data, policy, optimizer) into a training loop. BC
and PPO both ship today.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Policy(Protocol):
    sequence_length: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, N, C, H, W)`` latents; returns ``(B, 26)`` action."""
        ...


@runtime_checkable
class Algorithm(Protocol):
    """Marker protocol — concrete algorithms vary too much in surface area to
    constrain here. Each algorithm registers a class via
    :data:`nxrl.core.registry.algorithm_registry`; the launcher wires
    ``(model, dataloaders, config)`` to the algorithm's trainer.
    """
