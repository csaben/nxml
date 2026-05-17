"""Termination handling for Pokémon ZA.

nxrl's PPO rollout looks for ``__terminal__: True`` in the per-step reward
components dict; the pokemon_za reward factory injects that key when
``reward_shaper.detector.streak >= min_consecutive_hits`` (and
``unique_acquisition`` is disabled), so no extra wiring is needed.
"""
