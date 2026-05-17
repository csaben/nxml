"""Implementation of ``nxrl serve``."""

from __future__ import annotations


def run_serve(
    *,
    policy: str,
    port: int,
    host: str,
    transport: str,
    device: str | None,
    checkpoint_dir: str | None,
    enable_frame_mode: bool = False,
    vae_path: str | None = None,
) -> None:
    import torch

    from nxrl.serve.server import PolicyServer
    from nxrl.serve.transports.zmq import ZMQTransport

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    server = PolicyServer(
        model_path=policy,
        device=dev,
        checkpoint_dir=checkpoint_dir,
        enable_frame_mode=enable_frame_mode,
        vae_path=vae_path,
    )
    info = server.info()
    print(f"[nxrl serve] arch={info.architecture}  seq_len={info.sequence_length}  algo={info.algorithm}")
    print(f"[nxrl serve] device={dev}  transport={transport}  bind=tcp://{host}:{port}")
    if enable_frame_mode:
        print(f"[nxrl serve] frame mode ON  vae={vae_path or 'stabilityai/sd-vae-ft-mse'}  (single-client)")

    if transport != "zmq":
        raise NotImplementedError(f"transport {transport!r} not implemented")
    zt = ZMQTransport(server, port=port, host=host)
    try:
        zt.run()
    finally:
        zt.shutdown()
