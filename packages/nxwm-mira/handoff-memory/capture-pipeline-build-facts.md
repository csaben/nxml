---
name: capture-pipeline-build-facts
description: Ground-truth facts for building the HF spooler + collect UI upgrade + WebDataset loader (tasks 16-18)
metadata: 
  node_type: memory
  type: project
  originSessionId: 223ce196-8ffc-45c5-bdc2-b13851ec4661
---

Verified codebase facts for the capture-pipeline build (explored 2026-07-09). See [[nxwm-mira-reproduction]] for the why.

**Episode format (nxml-capture VideoParquetEpisodeWriter)**: per episode `{name}.mkv|.mp4` + `{name}.parquet` (frame_idx int64, timestamp f64, action fixed_size_list<f32,26>, zstd) + `{name}.manifest.json` written LAST (non-atomic; manifest presence = done signal). Video stream-encoded during capture; parquet buffered in RAM until close(). h264 profile = mp4/crf18/gop16; ffv1 = mkv lossless. `close()` with 0 frames returns None, writes nothing.

**nxml-collect** (apps/nxml-collect): one-process-one-episode; SIGINT sets cooperative stop flag; `--max-frames` hard cap; flat output dir, name = UTC `%Y%m%d_%H%M%S` (second-resolution → collision risk on fast auto-roll). `_stamp_metadata` rewrites manifest AFTER writer (adds game/orchestrator_url/camera_id). `--ui` = FastAPI on daemon thread sharing the V4L2 handle: GET / (static/index.html gamepad page), GET /mjpeg (multipart JPEG q70 ≤60fps), POST /action (httpx proxy → orchestrator HTTP), GET /health.

**nxml-autopilot has the auto-roll pattern to lift**: `recording.py::RecordingController` (thread-safe start/stop/append/status, fresh writer per episode) + `fresh_episode_path(root)` (per-episode SUBDIR, local-time `%Y%m%dT%H%M%S`) + web endpoints POST /recording/start|stop, GET /recording/status, token gate via X-Autopilot-Token header (`_check_token` in web.py). Autopilot samples `source.latest()` in its own 30Hz tick loop (no Synchronizer); collect's Synchronizer OWNS the frame loop — for mid-stream writer swap either wrap Synchronizer.frames() or adopt the tick model.

**nxbt-orchestrator API**: GET /health {running, connected, update_rate, override_window, recording} = ONLY connection-status source (WS /ws/state streams packets only, no status). POST /action {packet|vector, source: human|inference}, /buttons, /stick, /macro, /control. Pairing is BLOCKING at process start (`nx.wait_for_connection`) before the server accepts traffic — Switch must be on Change Grip/Order screen; success logs Switch MAC for `--reconnect-address` to skip future pairing. No pairing HTTP endpoint exists; UI pairing flow = poll /health + surface orchestrator lifecycle (systemd unit `nxbt-orchestrator.service`, root, py3.11 standalone uv tool — NOT in workspace).

**HF spooler is greenfield**: zero upload code in repo (only hf_hub_download in nxml-core/uri.py); webdataset not a dependency anywhere. Existing published dataset layout (data/za-mp4) is LeRobot-style (videos/chunk-NNN + meta/info.json + meta/episodes.parquet), not WebDataset — keep meta/info.json convention for dataset-level metadata if consistent.

**Deployment**: collect/autopilot = py3.14 workspace uv tools, run interactively today, no systemd units (spooler + long-running collect UI will be the first — write new units). Orchestrator = py3.11 system-python uv tool + systemd, needs root for BlueZ.
