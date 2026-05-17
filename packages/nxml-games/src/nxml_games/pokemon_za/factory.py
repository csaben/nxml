"""Factory adapting :class:`RewardShaper` to nxrl's ``Reward`` callable Protocol.

Wired into nxrl PPO via YAML::

    reward:
      callable: nxml_games.pokemon_za:make_reward_fn
      kwargs:
        config:                  # nested dataclass schema, see config.py
          start_frame_idx: 0
          goal_frame_idx: -1
          target_ui_detection:
            template_path: /path/to/bottom-right-ui.PNG
            min_consecutive_hits: 5
            ...
          locked_attack: {...}
          png_similarity_penalty:
            paths: [...]
            ...
        vae_path: stabilityai/sd-vae-ft-mse   # only needed when target_ui_detection
                                              # or png_similarity_penalty is set
        device: cuda

The adapter:
  - On ``info.step == 0`` resets shaper state and pulls ``start_latent`` /
    ``goal_latent`` from ``info["seed_latents"]`` using the config's
    ``start_frame_idx`` / ``goal_frame_idx`` (this is what makes the
    pokemon_za reward seed-aware without coupling nxrl's rollout to ZA
    semantics).
  - Adds ``__terminal__: True`` to the components dict when the
    target_ui-streak terminal condition fires.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import torch

from nxml_games.pokemon_za.config import RewardConfig
from nxml_games.pokemon_za.rewards import RewardShaper


def _dict_to_dataclass(cls, value: Any) -> Any:
    """Recursively convert a nested dict into the target dataclass.

    Handles ``Optional[Dataclass]`` (None passes through), ``list[Dataclass]``
    (element-wise conversion), nested dataclasses, and primitive fields.
    Dict-typed fields (``dict[str, StickTarget]``) get their values converted
    when the value type is a dataclass.
    """
    if value is None:
        return None
    if not is_dataclass(cls):
        return value
    if not isinstance(value, dict):
        return value

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in value:
            continue
        raw = value[f.name]
        ftype = f.type
        # Field annotations come back as strings in `from __future__ import
        # annotations` modules; resolve via the dataclass's actual __annotations__.
        annot = cls.__annotations__.get(f.name, ftype)
        kwargs[f.name] = _resolve(annot, raw)
    return cls(**kwargs)


def _resolve(annot: Any, raw: Any) -> Any:
    if isinstance(annot, str):
        # Lazy resolve: look up in pokemon_za.config namespace.
        import nxml_games.pokemon_za.config as _cfg_mod

        annot = eval(annot, dict(vars(_cfg_mod)))

    origin = get_origin(annot)
    args = get_args(annot)

    if origin is None and is_dataclass(annot):
        return _dict_to_dataclass(annot, raw)

    # Optional[X] -> Union[X, None]
    if origin is type(None):
        return None
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve(non_none[0], raw)

    if origin is list and args:
        elem_t = args[0]
        return [_resolve(elem_t, x) for x in (raw or [])]

    if origin is dict and len(args) == 2:
        _key_t, val_t = args
        return {k: _resolve(val_t, v) for k, v in (raw or {}).items()}

    return raw


def _maybe_load_vae(config: RewardConfig, vae_path: str | None, device: str | None):
    needs_vae = config.target_ui_detection is not None or (
        config.png_similarity_penalty is not None and config.png_similarity_penalty.paths
    )
    if not needs_vae:
        return None
    from nxwm.inference.vae import load_vae

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_vae(vae_path or "stabilityai/sd-vae-ft-mse", device=dev)


def make_reward_fn(
    *,
    config: dict[str, Any] | None = None,
    vae_path: str | None = None,
    device: str | None = None,
):
    """Factory entry point. ``config`` is the nested ``RewardConfig`` dict
    (typically loaded from YAML by nxrl's launcher). Returns a callable
    matching ``nxwm.env.rewards.protocol.Reward``.
    """
    cfg = _dict_to_dataclass(RewardConfig, config or {})
    if not isinstance(cfg, RewardConfig):
        raise TypeError(f"could not build RewardConfig from {config!r}")

    vae = _maybe_load_vae(cfg, vae_path, device)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    shaper = RewardShaper(cfg, vae=vae, device=dev)

    state: dict[str, Any] = {"initialized": False}

    def _reward(action: torch.Tensor, predicted_latent: torch.Tensor, info: dict[str, Any]):
        step = int(info.get("step", 0))
        n_frames = int(info.get("n_frames", 1))

        if step == 0 or not state["initialized"]:
            shaper.reset()
            state["attack_history"] = []
            seed_latents = info.get("seed_latents")
            if seed_latents is not None:
                start_idx = cfg.start_frame_idx
                if 0 <= start_idx < len(seed_latents):
                    shaper.start_latent = seed_latents[start_idx].unsqueeze(0).to(predicted_latent.device)
                if 0 <= cfg.goal_frame_idx < len(seed_latents):
                    shaper.goal_latent = (
                        seed_latents[cfg.goal_frame_idx].unsqueeze(0).to(predicted_latent.device)
                    )
            state["initialized"] = True

        reward, components = shaper.compute_reward(action, predicted_latent, step, n_frames)

        # Terminal selection: verified_lock_terminal (two-gate: streak +
        # attack-while-targeting) takes precedence if configured; otherwise
        # fall back to the streak-alone terminal. `vlt` + `vlt_*` component
        # keys are short for "Verified Lock Terminal" — brief so they fit
        # in the debug overlay panel.
        vlt = cfg.verified_lock_terminal
        if (
            vlt is not None
            and vlt.enabled
            and shaper.detector is not None
            and cfg.target_ui_detection is not None
        ):
            # Attack-press history (ring buffer of last attack_window steps),
            # gated on detector.streak > 0. The +8.0 button_logit_bias makes
            # attacks fire ~all the time, so only attacks pressed WHILE the
            # target popup is visible count toward verifying engagement.
            any_attack = any(action[0, idx].item() > 0.5 for idx in vlt.button_indices)
            seeing_target = shaper.detector.streak > 0
            atk_hist: list[int] = state["attack_history"]
            atk_hist.append(1 if (any_attack and seeing_target) else 0)
            if len(atk_hist) > vlt.attack_window:
                atk_hist.pop(0)

            streak_ok = shaper.detector.streak >= vlt.min_detector_streak
            attack_ok = sum(atk_hist) >= vlt.min_attack_presses

            # Surface the gate diagnostics so the debug overlay shows what's
            # missing for next frame's terminal eligibility.
            components["vlt_streak_ok"] = float(streak_ok)
            components["vlt_attack_ok"] = float(attack_ok)

            if streak_ok and attack_ok:
                components["__terminal__"] = True
                components["terminal_bonus"] = vlt.terminal_bonus
                final_reward = float(reward.item()) + vlt.terminal_bonus
                # Patch "total" so overlays / wandb see the bonus-inclusive value.
                components["total"] = final_reward
                return final_reward, components
        elif (
            cfg.unique_acquisition is None
            and shaper.detector is not None
            and cfg.target_ui_detection is not None
            and shaper.detector.streak >= cfg.target_ui_detection.min_consecutive_hits
        ):
            components["__terminal__"] = True

        return float(reward.item()), components

    # Attach the resolved config + the loaded VAE so callers (tests, debug
    # rendering, the trainer's eval-mkv hook) can introspect / reuse them.
    _reward.config = cfg  # type: ignore[attr-defined]
    _reward.vae = vae  # type: ignore[attr-defined]

    # Ensure asset paths exist; warn instead of crashing so tests with stub
    # configs still work.
    if cfg.target_ui_detection is not None:
        tpl = Path(cfg.target_ui_detection.template_path)
        if not tpl.exists():
            print(
                f"WARNING: target_ui template not found at {tpl} — detection will fire on shape-mismatched data."
            )

    return _reward
