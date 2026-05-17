"""Minimal Pre-LN encoder-only transformer.

This module exists as a *template* — its job is to demonstrate the steps
required for an architecture to show up in ``nxml-arch-viz``:

  1. Define an ``nn.Module`` here.
  2. Decorate it with ``@architecture_registry.register(name, config_cls=...)``.
  3. Ship a sibling ``spec.py`` exporting at minimum ``outer_spec()``.
  4. Import this package from ``nxwm/architectures/__init__.py`` so the
     decorator fires at import time.

It is *not* a world model — it does not implement ``init_rollout_state`` /
``step_rollout`` and won't plug into nxwm's training paths. A real world
model would also conform to the rollout protocol (see ``dit_v1`` for that).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nxwm.core.registry import architecture_registry

from .config import TinyTransformerConfig


class TinyTransformerBlock(nn.Module):
    """One Pre-LN transformer block: norm → attn → ⊕ → norm → mlp → ⊕."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


@architecture_registry.register("example_tiny", config_cls=TinyTransformerConfig)
class TinyTransformer(nn.Module):
    """Token-in → logits-out. ``forward(tokens) -> (B, T, vocab)``."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.embed_dim))
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(config.embed_dim, config.num_heads) for _ in range(config.depth)]
        )
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(tokens) + self.pos_embed[:, : tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(x)
