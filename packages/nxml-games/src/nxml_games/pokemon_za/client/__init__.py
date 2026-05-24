"""Live-inference client logic for Pokémon ZA.

Runtime decisions that aren't captured by the policy itself: when the
battle ends and we need to mash A through the post-battle dialogs, when
the Switch loses Bluetooth connection and a popup needs dismissing, etc.
Consumed by ``nxml-autopilot``'s trigger/macro system.
"""

from nxml_games.pokemon_za.client.connection_lost_detector import ConnectionLostDetector
from nxml_games.pokemon_za.client.death_state_machine import (
    DeathState,
    DeathStateMachine,
)
from nxml_games.pokemon_za.client.end_screen_detector import EndScreenDetector

__all__ = [
    "ConnectionLostDetector",
    "DeathState",
    "DeathStateMachine",
    "EndScreenDetector",
]
