"""Implementation of ``nxwm rollout``.

Drives a :class:`WorldModelClient` for ``--steps`` frames from a seed episode
and writes either:

  - ``--format gif``: the decoded frames as a GIF (works against in-process or
    remote — the bytes returned by ``client.step`` are JPEGs we just decode).
  - ``--format npz``: the raw scaled latents (in-process only — there's no
    transport-agnostic way to extract latents from ``step()`` over the wire).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _is_remote(model_uri: str) -> bool:
    return model_uri.startswith(("zmq://", "tcp://", "http://", "https://"))


def _load_actions(actions_path: str | None, *, steps: int, action_dim: int):
    import numpy as np

    if actions_path is None:
        return np.zeros((steps, action_dim), dtype=np.float32)
    arr = np.load(actions_path).astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != action_dim:
        raise ValueError(
            f"actions file must be (N, {action_dim}); got shape {arr.shape}"
        )
    if arr.shape[0] < steps:
        raise ValueError(
            f"actions file has {arr.shape[0]} rows but --steps {steps} requested"
        )
    return arr[:steps]


def _write_gif(frames_uint8, output: Path, *, fps: int = 10) -> None:
    """``frames_uint8``: list[ndarray (H, W, 3) uint8] → GIF."""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames_uint8]
    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def _decode_jpeg_to_rgb(jpeg_bytes: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode JPEG returned by server")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_rollout(
    *,
    model: str,
    seed_episode: str,
    start_frame: int,
    steps: int,
    output: str,
    out_format: str,
    device: str | None,
    flow_steps: int,
    cfg_scale: float,
    actions: str | None,
) -> None:
    import click
    import numpy as np

    from nxwm.serve import build_client

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path = Path(seed_episode).resolve()

    if out_format == "npz" and _is_remote(model):
        raise click.UsageError(
            "--format npz needs latent access; use a local/hf: model URI."
        )

    server_kwargs: dict[str, Any] = {
        "flow_steps": flow_steps,
        "cfg_scale": cfg_scale,
        "data_path": seed_path.parent,
        "load_vae_eagerly": out_format == "gif",
    }
    if device is not None:
        server_kwargs["device"] = device

    client = build_client(model, **server_kwargs)
    try:
        client.reseed(seed_path.name, start_frame=start_frame)

        # Need action_dim — pull from in-process or info().
        if hasattr(client, "server"):
            action_dim = client.server.model.action_dims  # type: ignore[attr-defined]
        else:
            info = client.info()
            action_dim = int(info["config"].get("action_dims", 26)) if "config" in info else 26
        action_arr = _load_actions(actions, steps=steps, action_dim=action_dim)

        click.echo(
            f"rollout: model={model} seed={seed_path.name}@{start_frame} "
            f"steps={steps} format={out_format} → {output_path}"
        )

        if out_format == "gif":
            frames_rgb = []
            for i in range(steps):
                jpeg = client.step(action_arr[i])
                frames_rgb.append(_decode_jpeg_to_rgb(jpeg))
            _write_gif(frames_rgb, output_path)
            click.echo(f"wrote {len(frames_rgb)} frames → {output_path}")
            return

        if out_format == "npz":
            # In-process only — pull latents directly from the underlying server.
            server = client.server  # type: ignore[attr-defined]
            latents = []
            for i in range(steps):
                lat = server.step_latent(action_arr[i])
                latents.append(lat.detach().cpu().numpy())
            np.savez(output_path, latents=np.stack(latents), actions=action_arr)
            click.echo(f"wrote {len(latents)} latents → {output_path}")
            return

        raise click.UsageError(f"unsupported format: {out_format}")
    finally:
        client.close()
