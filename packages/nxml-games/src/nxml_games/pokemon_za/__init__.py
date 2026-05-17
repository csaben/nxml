"""Pokémon ZA battle shaping for nxrl PPO.

Public surface — wired via YAML callable specs in ``configs/ppo/*.yaml``::

    reward:
      callable: nxml_games.pokemon_za:make_reward_fn
      kwargs: {...}
"""

from nxml_games.pokemon_za.action_spec import (
    ATTACK_BUTTON_INDICES,
    ATTACK_BUTTON_LOGIT_INDICES,
)
from nxml_games.pokemon_za.factory import make_reward_fn

__all__ = [
    "ATTACK_BUTTON_INDICES",
    "ATTACK_BUTTON_LOGIT_INDICES",
    "make_reward_fn",
]
