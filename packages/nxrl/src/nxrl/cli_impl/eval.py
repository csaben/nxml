"""Implementation of ``nxrl eval``."""

from __future__ import annotations

from pathlib import Path


def run_eval(
    *,
    policy: str,
    wm_path: str,
    seed_episode: str,
    start_frame: int,
    frames: int,
    goal_offset: int,
    flow_steps: int,
    cfg_scale: float,
    output: str,
    vae_path: str | None,
    device: str | None,
) -> None:
    import torch

    from nxrl.serve.server import PolicyServer
    from nxrl.training.eval_runner import generate_eval_gif

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load policy via PolicyServer (handles hf:/file: URIs and exposes the
    # post-processed action interface). Reuse it directly inside the runner.
    pserver = PolicyServer(model_path=policy, device=dev)
    print(f"[nxrl eval] policy={pserver.architecture} seq_len={pserver.sequence_length}")

    # Load world model.
    import nxwm.architectures  # noqa: F401  (registers DiT)
    from nxml_core.checkpoint import load_checkpoint
    from nxwm.core.registry import architecture_registry

    wm, _wm_cfg, _wm_ckpt = load_checkpoint(Path(wm_path), architecture_registry, device=dev)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    # Load VAE for decoding latents to RGB frames.
    from nxwm.inference.vae import load_vae

    vae = load_vae(vae_path or "stabilityai/sd-vae-ft-mse", device=dev)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = generate_eval_gif(
        policy_server=pserver,
        world_model=wm,
        vae=vae,
        seed_episode=Path(seed_episode),
        start_frame=start_frame,
        frames=frames,
        goal_offset=goal_offset,
        flow_steps=flow_steps,
        cfg_scale=cfg_scale,
        output_path=output_path,
        device=dev,
    )
    print(f"[nxrl eval] wrote {output_path} ({n_written} frames)")
