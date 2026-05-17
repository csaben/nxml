"""Implementation of ``nxrl rollout-debug``.

Runs the same PPO rollout loop the trainer uses, decodes each predicted
latent back to RGB, and writes one ``.mkv`` per (seed, rollout) with the
reward-component breakdown burned in as a side panel. Intended for
verifying that a reward stack is firing the way you expect on the
WM-generated frames — the wandb aggregates only tell you means, this
shows you which components fire where.

Reuses the production code paths via ``collect_rollout``'s new
``predicted_latents_out`` hook, so the policy/WM/reward path is bit-equal
to what the trainer runs each update.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

LATENT_SCALE = 0.18215

# Frame H, W after VAE decode (4, 16, 32) latents → 8x upscale = 128, 256 RGB.
FRAME_H = 128
FRAME_W = 256
FRAME_SCALE = 2  # blow up to 2x in the rendered canvas
SCALED_H = FRAME_H * FRAME_SCALE
SCALED_W = FRAME_W * FRAME_SCALE
PANEL_W = 320  # right-side text panel
CANVAS_W = SCALED_W + PANEL_W
CANVAS_H = SCALED_H

# Component display order — active-by-default first, fillers last so the eye
# lands on the rows most likely to be non-zero.
_COMPONENT_ORDER = [
    "total",
    "terminal_bonus",
    "target_ui",
    "unique_acq",
    "lock_atk",
    "movement",
    "png_pen",
    "slide_stag",
    "vlt_streak_ok",
    "vlt_attack_ok",
    "unstuck",
    "escape",
    "stuck_turn",
    "stick_pen",
    "action",
]


def _filter_to_dataclass(cls, raw: dict[str, Any]) -> dict[str, Any]:
    keys = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in keys}


def _load_policy(raw: dict[str, Any], policy_path: Path, device: torch.device, load_as: str):
    from nxrl.core.registry import policy_registry
    from nxrl import policies as _policies  # noqa: F401

    name = raw["policy"]["name"]
    cfg_cls = policy_registry.get_config(name)
    cfg = cfg_cls(**_filter_to_dataclass(cfg_cls, raw["policy"].get("config") or {}))
    policy_cls = policy_registry.get(name)
    policy = policy_cls(cfg).to(device)

    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if sd is None:
        raise ValueError(f"{policy_path} has no state_dict / model_state_dict")

    kind = load_as
    if kind == "auto":
        # PPO ckpt has the wrapper keys (log_std, button_bias, value_head...).
        kind = "ppo" if any(k.startswith("log_std") or k == "button_bias" for k in sd) else "bc"

    if kind == "ppo":
        policy.load_state_dict(sd)
    else:
        policy.load_bc_state_dict(sd)
    policy.eval()
    return policy, cfg, kind


def _load_world_model(raw: dict[str, Any], device: torch.device):
    import nxwm.architectures  # noqa: F401  (registers DiT)
    from nxml_core.checkpoint import load_checkpoint
    from nxwm.core.registry import architecture_registry

    wm_path = Path(raw["world_model"]["ckpt_path"])
    model, _cfg, _ckpt = load_checkpoint(wm_path, architecture_registry, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_reward_fn(raw: dict[str, Any], device: torch.device):
    import importlib

    spec = raw.get("reward") or {}
    if not spec.get("callable"):
        raise ValueError("config has no `reward.callable` — nothing to debug")
    module_name, fn_name = spec["callable"].split(":")
    mod = importlib.import_module(module_name)
    factory = getattr(mod, fn_name)
    kwargs = dict(spec.get("kwargs") or {})
    kwargs.setdefault("device", str(device))
    return factory(**kwargs)


def _load_vae(vae_path: str | None, device: torch.device):
    from nxwm.inference.vae import load_vae

    vae = load_vae(vae_path or "stabilityai/sd-vae-ft-mse", device=str(device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def _decode_latent_batch(latents: torch.Tensor, vae, device: torch.device) -> np.ndarray:
    """(N, 4, 16, 32) → (N, FRAME_H, FRAME_W, 3) uint8 RGB."""
    target_dtype = next(vae.parameters()).dtype
    z = latents.to(device, dtype=target_dtype) / LATENT_SCALE
    out = []
    # Decode one at a time — keeps peak VRAM low; this is debug code.
    for i in range(z.shape[0]):
        decoded = vae.decode(z[i : i + 1]).sample
        img = ((decoded + 1.0) / 2.0).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        out.append((img.float().cpu().numpy() * 255).astype(np.uint8))
    return np.stack(out, axis=0)


def _compute_bar_scales(components_per_step: list[dict[str, float]]) -> dict[str, float]:
    """For each component, the per-rollout max absolute value (used to scale bars).
    Falls back to 1.0 for components that are always zero."""
    scales: dict[str, float] = {}
    for key in _COMPONENT_ORDER:
        max_abs = max((abs(c.get(key, 0.0)) for c in components_per_step), default=0.0)
        scales[key] = max_abs if max_abs > 1e-6 else 1.0
    return scales


def _render_frame(
    frame_rgb: np.ndarray,
    components: dict[str, float],
    *,
    step: int,
    n_frames: int,
    bar_scales: dict[str, float],
    seed_label: str,
    terminal: bool,
) -> np.ndarray:
    """RGB (FRAME_H, FRAME_W, 3) + components → BGR (CANVAS_H, CANVAS_W, 3)."""
    # Upscale frame & convert to BGR for cv2 ops.
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_scaled = cv2.resize(
        frame_bgr, (SCALED_W, SCALED_H), interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:, :SCALED_W] = frame_scaled

    # Panel background — dark grey so text is readable.
    panel_x0 = SCALED_W
    canvas[:, panel_x0:] = (20, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header
    cv2.putText(
        canvas,
        f"step {step + 1:>4d}/{n_frames}",
        (panel_x0 + 8, 18),
        font,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if terminal:
        cv2.putText(
            canvas, "TERMINAL", (panel_x0 + 180, 18),
            font, 0.45, (60, 220, 60), 1, cv2.LINE_AA,
        )

    # Seed label (truncated if long).
    cv2.putText(
        canvas,
        seed_label[:34],
        (panel_x0 + 8, 36),
        font,
        0.36,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Component rows
    row_h = 14
    y0 = 58
    bar_x0 = panel_x0 + 140
    bar_w = PANEL_W - 148

    for i, key in enumerate(_COMPONENT_ORDER):
        val = components.get(key, 0.0)
        y = y0 + i * row_h

        # Highlight the total row.
        text_color = (220, 220, 220) if key != "total" else (180, 220, 255)
        cv2.putText(
            canvas, key, (panel_x0 + 8, y),
            font, 0.38, text_color, 1, cv2.LINE_AA,
        )
        # Value text (signed, 3 decimals).
        val_color = (
            (80, 220, 80) if val > 0
            else (80, 80, 220) if val < 0
            else (120, 120, 120)
        )
        cv2.putText(
            canvas, f"{val:+.3f}",
            (panel_x0 + 80, y),
            font, 0.38, val_color, 1, cv2.LINE_AA,
        )

        # Bar (centered around zero — left = negative, right = positive).
        scale = bar_scales.get(key, 1.0)
        norm = max(-1.0, min(1.0, val / scale)) if scale > 0 else 0.0
        center_x = bar_x0 + bar_w // 2
        cv2.line(canvas, (center_x, y - 4), (center_x, y), (90, 90, 90), 1)
        # Bar background
        cv2.rectangle(
            canvas,
            (bar_x0, y - 4),
            (bar_x0 + bar_w, y),
            (50, 50, 50),
            -1,
        )
        if norm > 0:
            cv2.rectangle(
                canvas,
                (center_x, y - 4),
                (center_x + int(norm * (bar_w // 2)), y),
                (60, 200, 60),
                -1,
            )
        elif norm < 0:
            cv2.rectangle(
                canvas,
                (center_x + int(norm * (bar_w // 2)), y - 4),
                (center_x, y),
                (60, 60, 200),
                -1,
            )
        # Center tick
        cv2.line(canvas, (center_x, y - 5), (center_x, y + 1), (130, 130, 130), 1)

    return canvas


def _write_mkv(frames_bgr: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {output_path} (mp4v codec). "
            "Try a .mp4 extension or rebuild opencv with x264 support."
        )
    for f in frames_bgr:
        vw.write(f)
    vw.release()


def render_rollout_frames(
    *,
    predicted_latents: list[torch.Tensor],
    reward_components: list[dict[str, float]],
    vae,
    seed_label: str,
    terminal: bool,
    device: torch.device | None = None,
) -> np.ndarray:
    """Decode latents + render overlay panel. Returns a ``(T, H, W, 3)`` BGR
    uint8 ndarray of rendered frames. No file IO."""
    if not predicted_latents:
        raise ValueError("predicted_latents is empty")
    if len(reward_components) != len(predicted_latents):
        raise ValueError(
            f"reward_components ({len(reward_components)}) != "
            f"predicted_latents ({len(predicted_latents)})"
        )
    dev = device or next(vae.parameters()).device
    latents_tensor = torch.stack(predicted_latents).squeeze(1)
    frames_rgb = _decode_latent_batch(latents_tensor, vae, dev)
    bar_scales = _compute_bar_scales(reward_components)
    rendered: list[np.ndarray] = []
    for i, frame in enumerate(frames_rgb):
        rendered.append(
            _render_frame(
                frame,
                reward_components[i],
                step=i,
                n_frames=len(predicted_latents),
                bar_scales=bar_scales,
                seed_label=seed_label,
                terminal=terminal and i == len(predicted_latents) - 1,
            )
        )
    return np.stack(rendered, axis=0)


def render_rollout_mkv(
    *,
    predicted_latents: list[torch.Tensor],
    reward_components: list[dict[str, float]],
    vae,
    output_path: Path,
    seed_label: str,
    terminal: bool,
    fps: int = 30,
    device: torch.device | None = None,
) -> Path:
    """Decode predicted latents to RGB frames, render overlay, write mkv via
    cv2 (mp4v codec). Local-debug oriented — mp4v plays in mpv/VLC but not
    inline in browsers. For wandb upload use :func:`render_rollout_frames`
    + ``wandb.Video(ndarray)`` instead (wandb re-encodes to H.264)."""
    frames = render_rollout_frames(
        predicted_latents=predicted_latents,
        reward_components=reward_components,
        vae=vae,
        seed_label=seed_label,
        terminal=terminal,
        device=device,
    )
    _write_mkv(list(frames), output_path, fps)
    return output_path


def run_rollout_debug(
    *,
    config_path: str,
    policy_path: str,
    output_dir: str,
    seed_indices: tuple[int, ...] | None,
    rollouts_per_seed: int,
    fps: int,
    device: str | None,
    load_as: str,
) -> None:
    import nxrl.algorithms  # noqa: F401  (registers PPO/BC algorithms)
    from nxrl.algorithms.ppo import RolloutSpec, SeedSpec
    from nxrl.algorithms.ppo.config import PPOAlgorithmConfig
    from nxrl.algorithms.ppo.rollout import collect_rollout, load_seed_episode
    from nxwm.inference.flow_matching import FlowMatchingSampler

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[rollout-debug] device={dev}")

    policy, _policy_cfg, kind = _load_policy(raw, Path(policy_path), dev, load_as)
    print(f"[rollout-debug] loaded policy from {policy_path} as kind={kind}")

    world_model = _load_world_model(raw, dev)
    print(f"[rollout-debug] loaded WM from {raw['world_model']['ckpt_path']}")

    vae_path = (raw.get("reward") or {}).get("kwargs", {}).get("vae_path")
    vae = _load_vae(vae_path, dev)

    reward_fn = _build_reward_fn(raw, dev)
    print("[rollout-debug] reward fn built")

    # Mirror trainer.button_logit_bias_indices wiring so the policy behaves
    # like it does during training (this is part of the policy's behavior,
    # not the optimizer's).
    algo_cfg_dict = (raw.get("algorithm") or {}).get("config") or {}
    algo_cfg = PPOAlgorithmConfig(**_filter_to_dataclass(PPOAlgorithmConfig, algo_cfg_dict))
    if algo_cfg.button_logit_bias_indices:
        with torch.no_grad():
            bias = policy.get_buffer("button_bias").clone()
            for idx in algo_cfg.button_logit_bias_indices:
                bias[idx] = algo_cfg.button_logit_bias_value
            policy.button_bias.copy_(bias)

    rollout_spec = RolloutSpec(**(raw.get("rollout") or {}))
    sampler = FlowMatchingSampler()

    all_seeds = [SeedSpec(**s) for s in (raw.get("seeds") or [])]
    if seed_indices is None or len(seed_indices) == 0:
        chosen = list(range(len(all_seeds)))
    else:
        chosen = list(seed_indices)

    bc_seq_len = policy.sequence_length
    out_root = Path(output_dir)

    for s_idx in chosen:
        seed = all_seeds[s_idx]
        seed_latents, seed_actions = load_seed_episode(seed)
        episode_stem = Path(seed.npz_path).stem
        seed_label = f"seed{s_idx:02d} {episode_stem}@{seed.start_frame}"

        for r_idx in range(rollouts_per_seed):
            preds: list[torch.Tensor] = []
            buf = collect_rollout(
                policy,
                world_model,
                sampler,
                seed_latents,
                seed_actions,
                config=algo_cfg,
                rollout=rollout_spec,
                reward_fn=reward_fn,
                device=dev,
                start_frame=seed.start_frame - bc_seq_len,
                predicted_latents_out=preds,
            )
            if not preds:
                print(f"  seed{s_idx:02d} roll{r_idx:02d}: 0 frames, skipping")
                continue

            print(
                f"  seed{s_idx:02d} roll{r_idx:02d}: {len(preds)} frames "
                f"reward_sum={sum(buf.rewards):+.3f} terminal={buf.terminal}"
            )

            out_path = (
                out_root
                / f"seed{s_idx:02d}_roll{r_idx:02d}_{episode_stem}_sf{seed.start_frame}.mkv"
            )
            render_rollout_mkv(
                predicted_latents=preds,
                reward_components=buf.reward_components,
                vae=vae,
                output_path=out_path,
                seed_label=seed_label,
                terminal=buf.terminal,
                fps=fps,
                device=dev,
            )
            print(f"    wrote {out_path}")

    print(f"[rollout-debug] done; outputs under {out_root}")
