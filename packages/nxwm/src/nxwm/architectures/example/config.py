from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TinyTransformerConfig:
    """Hyperparameters for the example tiny encoder-only transformer."""

    vocab_size: int = 256
    embed_dim: int = 64
    depth: int = 4
    num_heads: int = 4
    seq_len: int = 32
