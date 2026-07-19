---
name: nxwm-mira-reproduction
description: "Feasibility findings for reproducing MIRA world-model quality in nxwm (Pokémon ZA), hardware and dataset facts"
metadata: 
  node_type: memory
  type: project
  originSessionId: 223ce196-8ffc-45c5-bdc2-b13851ec4661
---

As of 2026-07-06, Clark wants to reproduce MIRA-quality world-model rollouts (references/mira, Rocket League, 5B latent diffusion) in packages/nxwm for Pokémon ZA, single-player only, with a browser player.

Key facts established:
- Local box "cradle": 2× RTX 3090 (24 GB each).
- nxwm dataset: 234 episodes, 641k frames @30fps ≈ 5.9 h, sd-vae-ft-mse latents (4,16,32) @128×256, per-frame 26-dim actions (nx-packets action_spec). HF: arelius/nxml-pokemon-legends-za-latents.
- MIRA dataset: ~29 TB, ~15.8k matches (~2,000 h) — data gap ~300× is the binding constraint.
- MIRA repo ships only a ~1B reference config (not the 5B demo model) and NO serving/web code; nxwm already has a gamepad browser player (serve/ + ui/).
- MIRA's transferable wins: DINOv3 RAE codec (vs SD-VAE — nxwm's weakest link), flow-matching DiT with kv-cached streaming, ~10-step inference. DINOv3 weights are Meta-gated.
- Past fa_dit run (~250M) reached epoch 4/200 in 5 days on one 3090 — recipe never converged.

Recommended path: record 50–100 h of a bounded ZA domain via nxml-capture, train a DINOv3-B/16 RAE codec, port MIRA WM at ~150–300M, reuse nxwm serve/UI.

Progress as of 2026-07-06 (codec phase DONE, training in flight):
- packages/nxwm-mira built: full RAEv2 codec port (codec/+ml/ near-verbatim, plain YAML+pydantic instead of Hydra, PIL-GIF wandb viz, live web viewer `nxwm-mira watch`). 41 tests in tests/nxwm_mira/. Configs at configs/codec/{tiny_smoke,za_3090}.yaml.
- DINOv3 weights downloaded (vitb16 + vitl16) at /home/arelius/models/dinov3; RS_DINO_WEIGHTS_DIR in repo .env.
- Key corpus facts: data/za-mp4 episodes mixed 480x640 + a few 720x1280 (dataset normalizes to 480x640); frames are 16:9 anamorphically squashed to 4:3 → encoder.aspect_mode: stretch.
- lpips works on py3.14; torch.compile off (torch 2.11 vs mira's 2.8 pin).
- FULL RUN in flight: 100k steps, 2×3090 torchrun, tmux `codec-train`, run dir checkpoints/codec/run_001, ~1.3 s/step → ~41 h ETA, wandb project nxwm-mira-codec-za, viewer tmux `codec-watch` :8800. GPU services paused via gpu-pause + interactive vLLM :8003 killed (restore: gpu-resume + `systemctl --user start vllm-qwen25vl`).
- WORLD MODEL PORT DONE (2026-07-06, overlapping with codec training): nxwm_mira/world_model/ (DiT + LatentWorldModel + streaming kv-cache inference, single-player only), Switch 26-dim ActionEncoder (sticks MLP + per-button embeddings) replacing keyboard+mouse, SwitchActions container in data/actions.py, ZAClipDataset returns pooled aligned actions (sticks mean / buttons OR over stride window). WM trains END-TO-END from video (frozen codec encodes per batch, mira-exact) — no latent pre-encoding. Trainer: training/wm_trainer.py, `nxwm-mira train-wm configs/wm-mira/za_3090.yaml` (~220M DiT, hidden 1024/16 layers/GQA, 32-frame windows, use_clean_past, betas 0.9/0.99 no decay). Play server has both modes: `nxwm-mira play --world-model CKPT` (real interactive dynamics, kv-cached) vs `--checkpoint` (codec roundtrip). 45 tests in tests/nxwm_mira/.
Data-collection scale-up plan (committed 2026-07-08, build AFTER WM run finishes):
- Capture topology: a separate Linux machine near the Switch does nxbt Bluetooth controller emulation + capture card, streams video to a web page; Clark (or anyone on the network) plays via browser gamepad. Cradle is too far for Bluetooth. "Jank but that's how it's been."
- Round-2 capture at 1280x720 native H.264 crf18 (~8-10 GB/h — fixes the anamorphic 4:3 squash; no more aspect_mode stretch). 100 h ≈ 1 TB → exceeds cradle's disk (1.8T, ~635G free), hence:
- Build (1) HF spooler on the capture machine: episode close → WebDataset tar shards (1-2 GB) → hf upload (Xet) → verify → delete local; (2) WebDataset streaming loader (MIRA's shuffle-buffer loader is the reference; our current ZAClipDataset is local-file random-access); (3) a clean pair+host Switch capture web UI — build as an UPGRADE of apps/nxml-collect --ui (which already does MJPEG stream + browser-gamepad→orchestrator forwarding + --max-frames cap): add in-UI nxbt pairing flow, long-running session with auto-roll at episode cap (today one process = one episode w/ Ctrl-C flush), spooler/upload/disk status strip, max-episode-length setting. Capture topology: Switch USB-C→HDMI capture card + nxbt Bluetooth on the same capture machine near the Switch. nxml-autopilot writes the same episode format (spooler serves both human and AI/hybrid sessions).

- PIPELINE COMPLETE (2026-07-09): codec run_001 finished 100k steps (final val L1 0.107/LPIPS 0.32); corpus encoded to data/mira-latents (233 eps, 1.1 GB); WM run_001 EARLY-STOPPED at checkpoint-100000 of 150k by Clark's decision (val diffusion plateaued ~0.27 from 25k on — the ~6h data ceiling). Final WM = checkpoints/wm-mira/run_001/checkpoint-100000 (298M). Play server (tmux wm-play, :8801) serves it interactively; watch on :8802. GPU services resumed (gpu-resume). One crash during WM training: PSU trip at full dual-GPU load (Jul 8 14:40, machine off till 22:57) — user's concurrent whisper request likely the marginal transient; power cap `sudo nvidia-smi -pl 280` recommended, never applied. Ops gotcha learned: pkill/pgrep -f patterns self-match the invoking shell — use [b]racket patterns.
- NEXT: build capture pipeline (tasks 16-18, see [[capture-pipeline-build-facts]]) — HF spooler, WebDataset loader, pair+host collect UI. Then Clark records 50-100h at 1280x720 and both models retrain.
