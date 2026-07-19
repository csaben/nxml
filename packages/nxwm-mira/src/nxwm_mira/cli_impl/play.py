"""`nxwm-mira play`: gamepad browser session over a codec checkpoint (nxwm UI)."""

from __future__ import annotations


def run_play(
    checkpoint: str | None,
    data_root: str,
    host: str,
    port: int,
    device: str | None,
    world_model: str | None = None,
    flow_steps: int = 10,
) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import uvicorn
    from nxwm.ui.app import build_app

    if world_model is not None:
        from nxwm_mira.play import WorldModelPlayClient

        client = WorldModelPlayClient(
            wm_checkpoint=world_model,
            data_root=data_root,
            device=device,
            n_diffusion_steps=flow_steps,
        )
        mode = "world-model (interactive dynamics)"
    else:
        from nxwm_mira.play import CodecPlayClient

        if checkpoint is None:
            raise SystemExit("Pass --checkpoint (codec) or --world-model (WM checkpoint).")
        client = CodecPlayClient(checkpoint=checkpoint, data_root=data_root, device=device)
        mode = "codec-roundtrip (actions captured, not yet driving dynamics)"

    app = build_app(client=client, game="pokemon-za")
    print(f"Gamepad session at http://{host}:{port}  [{mode}]")
    print(f"Model: {client.checkpoint_path}")
    print("Connect a controller, press any button, then Start/+ to toggle the play loop.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
