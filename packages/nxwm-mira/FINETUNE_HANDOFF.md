# Finetune handoff: codec + world model on a new ZA capture corpus

Self-contained brief for a fresh session. Goal: take the existing trained codec and world
model and finetune them on newly-collected Pokémon ZA data (larger, ideally 1280×720, ideally
a bounded domain like battles). Prior context: `~/.claude/.../memory/nxwm-mira-reproduction.md`
and `capture-pipeline-build-facts.md`.

## What already exists (do not rebuild)

- **Codec** `checkpoints/codec/run_001/checkpoint-99999` — RAEv2, DINOv3-B/16 frozen encoder +
  ~160M ViT decoder. Config: `configs/codec/za_3090.yaml`. Carries `latent_mean_std`.
- **World model** `checkpoints/wm-mira/run_001/checkpoint-100000` — 298M flow-matching DiT.
  Config: `configs/wm-mira/za_3090.yaml`. Trains **end-to-end from video** (frozen codec encodes
  each batch on the fly — there is NO separate latent dataset for WM training).
- **Package** `packages/nxwm-mira`: codec + world_model + trainers. `nxwm-mira {train, train-wm,
  encode, play, watch}`. Tests in `tests/nxwm_mira` (all green). Trainers auto-resume and support
  `run.continue_from` (full state) and `run.finetune_from` (weights only).
- **Spooler** `apps/nxml-spool`: new capture episodes → WebDataset tar shards → HF. Shard layout:
  `shard-NNNNNN.tar` with members `{episode_id}.mkv|.mp4`, `.parquet` (frame_idx/timestamp/
  action[26]), `.json` (manifest). Meta sidecars under `meta/`.

## The one prerequisite that gates everything: the data loader

The current training dataset, `ZAClipDataset` (`packages/nxwm-mira/src/nxwm_mira/data/za_dataset.py`),
reads the **local LeRobot-style layout** of `data/za-mp4/`: `meta/episodes.parquet` + video files +
`data/chunk-*.parquet` (per-frame actions). The new corpus is **WebDataset tar shards on HF** — a
different layout. So finetuning on the new data requires ONE of:

- **(A) Build the WebDataset streaming loader** (task #17, not yet built). Streams shards from HF,
  shard-shuffle + in-shard sequential decode + shuffle buffer (MIRA's loader is the reference).
  Must yield the same `(video_uint8_TCHW, actions_T26, ClipMeta)` items `collate_action_clips`
  expects, and pool source-fps actions onto sampled frames like `pool_actions` does. This is the
  clean path and also unblocks training at corpus scale. **Recommended.**
- **(B) Repack shards → local LeRobot layout** and reuse `ZAClipDataset` unchanged. A `tools/`
  script: download shards, extract, build `meta/episodes.parquet` + `data/chunk.parquet`. Faster to
  write, but needs the whole corpus on local disk (the thing the spooler existed to avoid).

Pick (A) unless the corpus is small enough to sit on disk.

## Finetune sequence

### 0. Prep
- New data collected + spooled to e.g. `arelius/nxml-pokemon-legends-za-v2`.
- `sudo nvidia-smi -pl 280` (PSU safety — see ops note in the report), then `/gpu-pause`.
- `uv sync --all-packages --extra training`; `RS_DINO_WEIGHTS_DIR` set (.env) for codec.

### 1. Codec finetune (~1 day, transfers well)
Copy `configs/codec/za_3090.yaml` → `configs/codec/za_v2.yaml` and change:
- `data.source` / loader → the new corpus (via the loader from the prerequisite above).
- `run.finetune_from: checkpoints/codec/run_001/checkpoint-99999` (weights-only; re-seeds latent EMA).
- If capturing true 1280×720 16:9: keep `encoder.video: {height: 288, width: 512}` but set
  `encoder.aspect_mode: pad` (720p is already 16:9 → pad is a no-op crop; `stretch` was only to
  undo the old corpus's anamorphic 4:3 squash). Verify with a quick recon.
- Fewer steps than from-scratch (e.g. 30–50k). Watch val LPIPS + recon GIFs on `nxwm-mira watch`.

Run: `torchrun --nproc-per-node=2 --module nxwm_mira.cli train configs/codec/za_v2.yaml
--output-dir checkpoints/codec/run_002` in tmux. Produces a NEW `latent_mean_std`.

> **If you skip the codec finetune** (codec unchanged), you can go straight to the WM finetune
> and keep `codec_checkpoint: checkpoints/codec/run_001`. Only finetune the codec if 720p / new
> visual domain materially shifts the pixels.

### 2. World model finetune (~2–4 days at 298M)
Copy `configs/wm-mira/za_3090.yaml` → `configs/wm-mira/za_v2.yaml` and change:
- `model.config.codec_checkpoint` → the codec you're using (`run_002` if you finetuned it, else `run_001`).
  **Critical:** if the codec changed, the latents changed — the WM MUST be finetuned against the
  new codec, not the old one. The WM reads `latent_mean_std` from whatever codec it points at.
- `run.finetune_from: checkpoints/wm-mira/run_001/checkpoint-100000` (only valid if hidden_dim /
  n_layers / n_head are unchanged — a resize means training a fresh WM, not finetuning).
- data → new corpus loader.
- If bumping model size (e.g. hidden 1024→1536 for a cloud burst), DROP `finetune_from` — it's a
  new run.

Run: `torchrun --nproc-per-node=2 --module nxwm_mira.cli train-wm configs/wm-mira/za_v2.yaml
--output-dir checkpoints/wm-mira/run_002` in tmux. Rollout GIFs every `viz_every` on `watch` :8802.

### 3. Serve + eval
- `nxwm-mira play --world-model checkpoints/wm-mira/run_002 --host 0.0.0.0 --port 8801` (tmux).
  Xbox controller in browser is the real eval. Hot-swap checkpoints from the reload box mid-run.
- Watch the orange-bordered region of rollout GIFs for how far coherence holds vs ground truth.
- `/gpu-resume` when training ends.

## Gotchas (learned the hard way)
- **Config layout**: this package writes `model.config`; loaders accept both that and MIRA's
  `model.architecture.config`. Codec `latent_mean_std` lives in the checkpoint `extra_data`.
- **`finetune_from` vs `continue_from`**: finetune = weights only, fresh optimizer/step-0 (use for
  new data). continue = full state resume (use for crash recovery). Both in `run.*`.
- **Mixed resolutions**: `ZAClipDataset` normalizes to a `frame_hw` so batches stack; keep that in
  the new loader (720p + any stragglers).
- **Ops**: `pkill -f` / `pgrep -f` self-match the invoking shell — use `[b]racket` patterns. tmux
  keeps runs alive across sessions; watch pages are file-based (status.json/metrics.jsonl/recons).
- **Everything is uncommitted** in the working tree as of this writing — `git status` before you
  start so you know your baseline.

## Success criterion
50–100 h of bounded-domain data → target ~10 s coherent interactive generations in-domain. On the
old ~6 h corpus the WM plateaued at ~1–3 s (data ceiling, not a code bug). The finetune's payoff is
proportional to how much more (and how much narrower) the new data is.
