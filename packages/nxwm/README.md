# nxwm

World model training, inference, and serving for Nintendo Switch games. Encode
video episodes to latents via a VAE, train a diffusion-based world model on
those latents, then serve rollouts via CLI or a UI for probing.

## Install

From a workspace checkout: `uv sync --all-packages --all-extras`, then `uv run nxwm …`.

Standalone (no clone), as a `uv tool`:

```bash
uv tool install --with-extras training \
    "git+https://github.com/csaben/nxml.git#subdirectory=packages/nxwm"

# Pin to a commit or tag for reproducibility:
#   "git+https://github.com/csaben/nxml.git@<sha>#subdirectory=packages/nxwm"

# Skip wandb logging (drop the extras):
#   uv tool install \
#       "git+https://github.com/csaben/nxml.git#subdirectory=packages/nxwm"
```

The `training` extras pull in `wandb` + `python-dotenv`. `nxwm train` logs
runs to wandb automatically once you `wandb login` and your config's
`training.project_name` is set (the bundled configs already set it).

uv clones the whole repo so the workspace siblings (`nxml-core`,
`nx-packets`) resolve, and the `pytorch-cu128` index declared in
`pyproject.toml` is honored for the torch wheels. Once installed, drop
the `uv run` prefix from the invocations below.

## Quick start

```bash
# 1. VAE-encode episodes to latents (NVDEC if CUDA available, else CPU).
nxwm encode --input data/episodes --output data/latents

# 2. Train a world model (the bundled tiny config is a CPU-friendly smoke test).
nxwm train configs/wm/wm_dit_v1_tiny.yaml

# 3. Visualize a checkpoint against held-out episodes.
#    Grab a trained example from HuggingFace if you don't have your own yet:
nxwm ui pokemon_za \
    --model hf:arelius/nxwm-pokemon-za/run_021_best.pt \
    --data-path data/latents
```

## Usage

### Train on the public dataset

If you don't have your own recorded episodes yet, the Pokémon ZA latents
used in the bundled checkpoints are published at
[`arelius/nxml-pokemon-legends-za-latents`](https://huggingface.co/datasets/arelius/nxml-pokemon-legends-za-latents).
The dataset ships as parquet chunks; `tools/latents_parquet_to_npz.py`
splits them back into the per-episode `.npz` shape
`nxwm.training.dataset.LatentEpisodeDataset` consumes.

```bash
# 1. Download the dataset.
hf download arelius/nxml-pokemon-legends-za-latents \
    --repo-type dataset --local-dir ./za-latents

# 2. Split the parquet chunk(s) into per-episode .npz files.
uv run --with click --with pandas --with pyarrow python \
    tools/latents_parquet_to_npz.py \
    --parquet ./za-latents/data/chunk-000.parquet \
    --meta    ./za-latents/meta/episodes.parquet \
    --out     ./data/latents

# 3. Train. The bundled za config points at ./data/latents out of the box.
nxwm train configs/wm/wm_dit_v1_za.yaml
```

Bundled configs (all point at `./data/latents` out of the box; just swap
the YAML to compare architectures):

- `wm_dit_v1_tiny.yaml` — CPU smoke run (depth-2 / 2 epochs).
- `wm_dit_v1_za.yaml` — conservative dit_v1 baseline (depth-6 / embed-256,
  50 epochs, single-GPU friendly).
- `wm_fa_dit_za.yaml` — Future-Anchored DiT (paper:
  [arXiv:2504.19077](https://arxiv.org/abs/2504.19077)) with anchor-frame
  conditioning and a multi-hypothesis plan head. Bigger model (~250M),
  longer training (200 epochs); same dataset and `nxwm train` invocation.

For larger runs copy any of these and bump the architecture and
`data.epochs` to taste.

### Offline rollout

Generate frames offline from a model + seed episode. Decode as a GIF
(works in-process or remote) or save raw scaled latents (in-process only):

```bash
nxwm rollout --model hf:arelius/nxwm-pokemon-za/run_021_best.pt \
             --seed-episode data/episodes/run_001.npz \
             --start-frame 100 --steps 60 \
             --output ./rollout.gif

nxwm rollout --model ./checkpoints/best.pt \
             --seed-episode data/episodes/run_001.npz \
             --output ./rollout.npz --format npz
```

### Probe with a target-UI detector

```bash
nxwm ui pokemon_za \
    --model hf:arelius/nxwm-pokemon-za/run_021_best.pt \
    --data-path data/latents \
    --detector pokemon_za:target_ui \
    --detector-arg template_path=path/to/template.PNG
```

`template_path` is required — point it at any grayscale image of the UI
element you want to detect (e.g., the move-selection box for Pokémon ZA).

Detector tuning that's worked well on Pokémon ZA:

```yaml
target_ui_detection:
  score_threshold: 0.1
  sat_threshold: 63
  min_consecutive_hits: 3
```

### Serve a world model

```bash
# ZMQ transport (the wire format the UI uses by default).
nxwm serve --model ./checkpoints/best.pt --transport zmq --port 5556

# HTTP/WebSocket transport (LAN, no TLS):
nxwm serve --model ./checkpoints/best.pt --transport http \
           --host 0.0.0.0 --port 8000

# HTTPS (TLS terminated in-process):
nxwm serve --model ./checkpoints/best.pt --transport http \
           --host 0.0.0.0 --port 8443 \
           --ssl-certfile /etc/letsencrypt/live/host/fullchain.pem \
           --ssl-keyfile  /etc/letsencrypt/live/host/privkey.pem
```

### Web UI

```bash
nxwm ui my-game --model ./checkpoints/best.pt          # in-process, local file
nxwm ui my-game --model hf:owner/repo/file.pt          # in-process via HF
nxwm ui my-game --model zmq://gpu-box.local:5556       # remote ZMQ server
nxwm ui my-game --model https://gpu-box.example:8443   # remote HTTP server
nxwm ui my-game --model ... --host 0.0.0.0 --no-browser
```

### `--model` URI dispatch

One flag, three transports (in-process, ZMQ, HTTP) — `nxwm` picks based on the URI scheme:

| Scheme                   | Resolver                          | Transport                |
| ------------------------ | --------------------------------- | ------------------------ |
| `./path`, `/abs/path`    | local path                        | in-process               |
| `hf:owner/repo/file.pt`  | `huggingface_hub.hf_hub_download` | in-process               |
| `zmq://host:port`        | (rewrites to `tcp://`)            | `RemoteZMQClient`        |
| `tcp://host:port`        | direct                            | `RemoteZMQClient`        |
| `http://host[:port]`     | FastAPI + WebSocket session       | `RemoteHTTPClient`       |
| `https://host[:port]`    | same, with TLS                    | `RemoteHTTPClient`       |

In-process flags (`--device`, `--flow-steps`, `--cfg-scale`, `--data-path`) are
ignored when the URI is remote — the server is expected to have been
configured at launch.

### Deployment shapes

```
1. All-in-one (development, single GPU machine):
     nxwm ui my-game --model ./checkpoints/best.pt

2. Split (UI on laptop, GPU box on LAN):
     gpu-box$  nxwm serve --model ./checkpoints/best.pt --host 0.0.0.0
     laptop$   nxwm ui my-game --model zmq://gpu-box.local:5556

3. Hybrid (model behind any zmq:// endpoint, e.g. cloud):
     same as #2 with a different endpoint URL in --model.
```

## CLI

```
nxwm encode   --input DIR --output DIR    [--device cuda|cpu] [--vae URI] [--fps N]
nxwm train    CONFIG                      [--resume PATH] [--world-size N]
nxwm rollout  --model URI --seed-episode NPZ --output PATH
              [--start-frame N] [--steps N] [--format gif|npz]
nxwm serve    --model URI [--transport zmq|http] [--host H] [--port N]
              [--ssl-certfile PATH] [--ssl-keyfile PATH]
nxwm ui       GAME --model URI [--data-path DIR] [--detector NAME]
              [--detector-arg KEY=VAL ...] [--port N] [--host H] [--no-browser]
```

---

## Development

### Notes

- **Color order.** `nxwm encode` expects BGR HWC uint8 (cv2-native), the
  same shape `nxml-capture`'s writers produce.
