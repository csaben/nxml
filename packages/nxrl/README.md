# nxrl

Behavior cloning + reinforcement learning for Nintendo Switch policies.

- **BC**: `Policy` + `Algorithm` protocols + registries. Architectures
  `bc_transformer_v1` and `bc_lstm_v1`. Stick-MSE + button-BCE loss with
  action-weighting, DDP, wandb, self-describing checkpoints via
  `nxml_core`.
- **PPO**: `ppo_policy_v1` (BC base + value head + learnable per-stick
  `log_std` + button-bias buffer). Clipped surrogate, value loss, entropy
  bonus, BC anchor against a frozen reference. Single-process rollouts
  driving any registered `nxwm.WorldModel` via `step_rollout` +
  `FlowMatchingSampler`.
- **Serve + eval**: `PolicyServer` + ZMQ transport (stateless: latents-in
  / 26-dim action-out). `PolicyClient` Protocol with `InProcessClient` and
  `RemoteZMQClient`; `build_client(uri)` URL-scheme dispatcher. `nxrl
  eval` rolls a policy through a frozen WM and writes a GIF.

The 13-component pokemon_za reward stack and game-specific client logic
live in `nxml-games/pokemon_za/`, never in `nxrl` — the dependency goes
one way.

## Install

From a workspace checkout, just `uv sync` and use `uv run nxrl …`.

Standalone (no clone), as a `uv tool`:

```bash
uv tool install \
    "git+https://github.com/csaben/nxml.git#subdirectory=packages/nxrl"

# Pin to a commit or tag for reproducibility:
#   "git+https://github.com/csaben/nxml.git@<sha>#subdirectory=packages/nxrl"
```

uv clones the whole repo so the workspace siblings (`nxml-core`,
`nx-packets`, `nxwm`) resolve. Once installed, drop the `uv run` prefix
from the invocations below.

## Quick start

```bash
uv sync --all-packages --all-extras
uv run nxrl train configs/bc/seq300_transformer.yaml
```

## CLI

```bash
nxrl train CONFIG       # BC or PPO, dispatched by config's algorithm.name
nxrl serve --policy URI [--port 5557] [--device cuda] [--enable-frame-mode]
nxrl eval --policy URI --world-model PATH --seed-episode NPZ \
          --frames 150 --output OUT.gif
nxrl rollout-debug --config CONFIG --policy CKPT --output-dir DIR
          # PPO per-seed eval, writes per-rollout .mkv files
```

## Usage

### Train BC

```bash
# Edit configs/bc/seq300_transformer.yaml: data.data_paths / val_files.
uv run nxrl train configs/bc/seq300_transformer.yaml

# Resume from a saved checkpoint.
uv run nxrl train configs/bc/seq300_transformer.yaml \
    --resume checkpoints/bc/run_001/best.pt
```

### Train PPO

```bash
# Edit configs/ppo/escape_seeds.yaml: world_model.ckpt_path,
# bc_init.ckpt_path, seeds[*].npz_path.
uv run nxrl train configs/ppo/escape_seeds.yaml

# Resume from a saved checkpoint.
uv run nxrl train configs/ppo/escape_seeds.yaml \
    --resume checkpoints/ppo/run_001/update_0002.pt
```

### Serve a policy

```bash
# Bind a policy server on tcp://*:5557
uv run nxrl serve --policy checkpoints/bc/run_001/best.pt --port 5557
```

**Frame mode.** With `--enable-frame-mode` the server holds **both** the
VAE and the sliding latent window; clients ship a single JPEG per tick
(~5 KB) instead of a full latent stack. Use this when the play box has no
GPU. Single-client by construction (one window per server).

```bash
uv run nxrl serve --policy checkpoints/bc/run_001/best.pt --port 5557 \
    --enable-frame-mode
```

From Python:

```python
from nxrl.serve import build_client

client = build_client("zmq://gpu-box.local:5557")
info = client.info()                 # arch, seq_len, latent_shape, action_dim
action = client.predict(latents)     # latents: (T, 4, 16, 32) float32
                                     # returns (26,) float32: sticks[:4] + buttons[4:]
client.reload("/path/to/new.pt")     # hot-swap the model on the running server
```

`build_client` dispatches:

- `zmq://host:port` / `tcp://host:port` → `RemoteZMQClient`.
- `hf:owner/repo/file.pt` / file path → `InProcessClient` wrapping a local
  `PolicyServer`.
- `http://` / `https://` → `NotImplementedError` (stub).

### Generate an eval GIF

```bash
uv run nxrl eval \
    --policy        checkpoints/bc/run_001/best.pt \
    --world-model   checkpoints/wm/wm_dit_v1.pt \
    --seed-episode  data/latents/episode.npz \
    --start-frame   0 \
    --frames        150 \
    --goal-offset   30 \
    --output        rollouts/run_001_seed00.gif
```

The runner zero-pads the context for negative `start_frame`, advances a
receding goal each step, denoises with `FlowMatchingSampler`, decodes
latents through the VAE, stitches a 30 fps GIF.

---

## Development

### Wire protocol (ZMQ)

| Mode             | Byte   | Payload                                                                      | Response                                                                |
| ---------------- | ------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `PREDICT`        | `0x00` | `<u32 T><u32 C><u32 H><u32 W>` + `T*C*H*W*4` float32 bytes                   | 104-byte action (26 float32)                                            |
| `PREDICT_FRAME`  | `0x01` | raw JPEG bytes (one frame); only valid when server started with `--enable-frame-mode` | 104-byte action, or empty bytes while the server's window is warming up |
| `RELOAD`         | `0x03` | JSON `{"model_path": str}`                                                   | `b"OK"`                                                                 |
| `INFO`           | `0x04` | (none)                                                                       | JSON: arch, sequence_length, latent_shape, action_dim, algorithm        |

Errors: `b"ERROR: <msg>"`. Exceptions never escape the server loop.
