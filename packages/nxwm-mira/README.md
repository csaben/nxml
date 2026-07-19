# nxwm-mira

Port of [MIRA](https://github.com/mira-wm/mira)'s RAEv2 codec (representation
autoencoder) for Nintendo Switch world models: a frozen DINOv3 encoder + ViT
video decoder trained with L1 + LPIPS + DINO-consistency losses on raw gameplay
episodes. Replaces the generic SD-VAE latents `nxwm` trained on. The world-model
port comes next; codec checkpoints already carry everything it needs
(`latent_mean_std`, discoverable `codec_config.yaml`).

## Setup

```bash
uv sync --all-packages --extra training   # wandb + lpips (+ python-dotenv)
```

Real training needs the Meta-gated DINOv3 weights
(`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` from the
[DINOv3 downloads page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)).
Point `RS_DINO_WEIGHTS_DIR` at their directory (a repo-root `.env` works — the
train CLI loads it). Smoke runs (`require_dino_weights: false`) skip this.

## Usage

```bash
# CPU smoke test (fake data, random backbone, ~2 min)
uv run nxwm-mira train configs/codec/tiny_smoke.yaml

# Single GPU on the ZA corpus (data/za-mp4)
uv run nxwm-mira train configs/codec/za_3090.yaml

# Both GPUs (explicit --output-dir so ranks agree)
uv run torchrun --nproc-per-node=2 --module nxwm_mira.cli \
    train configs/codec/za_3090.yaml --output-dir checkpoints/codec/run_001

# Live training viewer (file-based; works mid-run, post-run, over the network)
uv run nxwm-mira watch checkpoints/codec/run_001 --host 0.0.0.0 --port 8800
```

Runs auto-resume: relaunching with the same output dir continues from the
latest checkpoint and re-attaches the same wandb run (`wandb_run_id.txt`
sidecar). `--resume` / `run.finetune_from` cover the explicit cases.

## What the trainer writes

```
run_dir/
├── codec_config.yaml       # discoverable by VideoCodec.load_from_checkpoint
├── checkpoint-{step}/      # checkpoint.pth (EMA weights + latent_mean_std)
│                           # + training_state.pth (optimizer/scheduler/EMAs)
├── status.json             # atomic; read by `nxwm-mira watch`
├── metrics.jsonl           # one line per log event
├── recons/step_*.gif       # GT|recon side-by-sides (also logged to wandb
│                           #   as videos/reconstruction every run.viz_every)
└── wandb_run_id.txt        # crash-resume re-attach
```

## Loading a trained codec

```python
from nxwm_mira.codec.codec_model import VideoCodec

codec = VideoCodec.load_from_checkpoint("checkpoints/codec/run_001/checkpoint-99999/checkpoint.pth")
mean, std = codec.info_from_checkpoint["latent_mean_std"]  # world-model latent normalization
input_video, enc = codec.encode(video)   # video: (B, T, 3, H, W) in [-1, 1] after preprocess
decoded = codec.decode(enc.z)            # tanh-bounded [-1, 1]
```

MIRA-written checkpoints load too (the config loader accepts both layouts).

## Notes

- **Aspect mode.** The ZA corpus stores 16:9 content anamorphically squashed
  into 4:3 frames; `encoder.aspect_mode: stretch` un-squashes during
  preprocessing. The default `pad` preserves MIRA-checkpoint behavior.
- **Mixed resolutions.** Episodes at 720×1280 are normalized to the dominant
  480×640 inside the dataset so batches stack.
- **torch.compile stays off** (`run.compile: false`): MIRA pins torch 2.8 for
  inductor; this workspace runs 2.11.
- Upstream: `references/mira` (Apache 2.0). `codec/` and `ml/` are near-verbatim
  ports; the trainer drops Hydra for plain YAML + pydantic.
