"""Pokémon ZA-specific action constants.

The 26-dim action vector itself is defined by ``nx_packets`` and is
game-agnostic. What's game-specific is the *meaning* of certain buttons
in this game (Y/X/B/A are the four attack buttons in ZA battles) and the
indices PPO regularization configs reach for when biasing exploration
toward attacks.
"""

from __future__ import annotations

from typing import Final

from nx_packets import BUTTON_INDEX

# Absolute indices into the 26-dim action vector for the four attack buttons.
ATTACK_BUTTON_INDICES: Final[list[int]] = [
    BUTTON_INDEX["Y"],
    BUTTON_INDEX["X"],
    BUTTON_INDEX["B"],
    BUTTON_INDEX["A"],
]

# Indices into the 22-dim raw button head (action_index - 4). Used by
# ``ppo.button_logit_bias_indices`` in the PPO algorithm config to add a
# constant exploration bias to attack-button logits.
ATTACK_BUTTON_LOGIT_INDICES: Final[list[int]] = [i - 4 for i in ATTACK_BUTTON_INDICES]
