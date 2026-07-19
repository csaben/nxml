# nxml-spool

Capture-machine spooler: watches episode output dirs, packs completed episodes
(video + actions/events parquet + manifest) into ~1 GB WebDataset tar shards,
publishes them through an abstract storage backend, verifies an immutable
checksum commit, then deletes the local files. Hugging Face dataset repos
remain supported; mounted cluster/object storage avoids coupling ingest to Git.
Local disk becomes a sliding buffer instead of an archive.

Works with both capture layouts: `nxml-collect` (flat files) and
`nxml-autopilot` web mode (one subdirectory per episode). An episode counts as
complete when its `.manifest.json` exists (writers emit it last) and nothing
has been modified for `--settle-seconds`.

## Usage

```bash
# Auth once: `hf auth login` (or HF_TOKEN env).
nxml-spool run \
    --watch ~/captures/za \
    --repo arelius/nxml-pokemon-legends-za-v2 \
    --shard-size-mb 1024

nxml-spool status         # reads status.json from the state dir
nxml-spool run ... --once # single pass (seal + ship whatever is ready)
nxml-spool run ... --no-delete  # dry-ish run: upload but keep local files

# Mounted S3/cluster storage instead of Hugging Face:
nxml-spool run --watch ~/captures/za --repo unused/local \
    --storage-root /mnt/nxml-objects
```

Below-size shards are sealed anyway once their oldest episode has waited
`--flush-partial-after` (default 15 min), so a short capture session still
becomes durable promptly.

## State & crash safety

`--state-dir` (default `~/.local/state/nxml-spool`) holds `journal.json`
(episodes shipped, next shard index — atomically rewritten), `staging/`
(shards being built), and `status.json` (live stats for the capture UI:
pending episodes, current upload, disk headroom). Episodes are journaled as
shipped only after the backend verifies the tar checksum and publishes its
commit marker. A crash after remote commit but before the local journal is
safe: restart republishes the same immutable bytes, observes the existing
matching marker, then completes the journal transition. Conflicting bytes are
never overwritten.

Disk pressure uses hysteresis: `--disk-high-watermark` (default 0.85) enters
pressure mode and forces partial shards to seal immediately; it clears only
below `--disk-low-watermark` (default 0.75). `status.json` exposes the current
state. Source episodes are never deleted merely to satisfy a watermark;
deletion remains gated by a durable checksum commit.

## systemd (capture machine)

```ini
# ~/.config/systemd/user/nxml-spool.service
[Unit]
Description=nxml episode spooler -> HuggingFace
After=network-online.target

[Service]
ExecStart=%h/.local/bin/nxml-spool run --watch %h/captures/za --repo arelius/nxml-pokemon-legends-za-v2
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

Install: `uv tool install "git+https://github.com/csaben/nxml.git#subdirectory=apps/nxml-spool"`,
then `systemctl --user enable --now nxml-spool` (enable lingering so it runs
without a login session).
