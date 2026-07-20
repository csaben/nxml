# Codec domain-shift audit operation

This gate measures whether the frozen `checkpoint-99999` codec materially
degrades on eligible compact gameplay. It does not train, submit a job, or make
a model eligible for deployment.

The immutable `nxml.codec-domain-shift-input.v1` record must be written before
evaluation. It names the cluster snapshot and pinned Hub revision; explicitly
selected legacy and compact episode IDs/splits and their passing dispositions;
the checkpoint ID, checkpoint SHA-256, and original config SHA-256; artifact-
probed compact codec lineage; the audit seed, Git commit, and config SHA-256; and
all metric thresholds, confidence level, and minimum windows. Missing compact
gameplay, human ownership, motion, codec evidence, or quality eligibility fails
validation. Static, neutral, contaminated, canary, and vetoed episodes cannot be
selected.

`deterministic_windows` hashes the seed, episode ID, and exact global frame
ordinals to select reproducible MIRA windows. Preprocessing resizes isotropically
and pads symmetrically to the declared frame size. It never stretches or crops.
Each reconstruction is evaluated with pixel L1, LPIPS, DINO cosine distance,
temporal motion error, and action-weighted temporal error. The metric runner must
use the frozen checkpoint named in the input record and store representative
first/middle/last ground-truth-versus-reconstruction PNGs by SHA-256.

The decision compares matched legacy and compact window distributions. A metric
is considered materially shifted only when the lower bound of its deterministic
1,000-resample bootstrap relative-increase interval exceeds its predeclared
threshold. Any exceeded metric produces `finetune_codec`; otherwise the record
produces `retain_codec`. Insufficient samples produce no decision. The complete
decision record is content-addressed and includes every sample and artifact ID;
no placeholder or inferred measurement is accepted.

A subsequent `nxml.codec-job.v1` manifest is declarative only. A new-domain run
uses `mode=finetune_from`, checkpoint `checkpoint-99999`, fresh optimizer state,
and at most 50,000 steps. `continue_from` requires the exact prior job ID and is
only crash recovery for that same run. Creating this manifest does not authorize
starting it.

After inspector deployment, collect the six infrastructure-only codec probes:

```bash
uv run python tools/inspect_compact_canary_segments.py /tmp/compact-inspections.json
```

The runner reads the existing token locally, never prints it, validates every
response against `nxml.compact-h264-720p60.v1`, and writes only evidence. These
episodes remain vetoed and are not valid audit samples.

Before any real publication, run the offline eligibility check against an
authoritative snapshot and its inspection set:

```bash
uv run python tools/webdataset_publication_dry_run.py \
  /path/to/snapshot.json /path/to/inspections.json EPISODE_ID...
```

It performs no upload. It rejects excluded/vetoed episode IDs, IDs absent from
the eligible snapshot, missing inspections, decoded/action mismatch, and any
codec incompatibility.
