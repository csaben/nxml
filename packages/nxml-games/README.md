# nxml-games

Game-specific code: reward shaping, terminations, seeds, presets, and live
inference client logic. Each game lives under its own subdirectory and is
designed to be mechanically extractable into a standalone
``nxml-env-<game>`` PyPI package once the in-repo contracts are proven.

The critical invariant: **nxwm and nxrl never import from nxml-games**.
Game code is consumed via a callable spec in YAML configs (e.g.,
``reward.callable: nxml_games.pokemon_za:make_reward_fn``) so the
dependency graph stays one-way.

## Games

- ``pokemon_za`` — Pokémon ZA battle shaping. 13-component RewardShaper +
  TargetUIDetector + battle-end termination + 12-seed battle-start
  curriculum + preset. The live-inference client logic (end-screen detector,
  connection-lost detector, death state machine) lives under
  ``pokemon_za/client/``; live trigger-driven autobattling ships inside
  ``nxml-autopilot`` (Triggers + MashController).
