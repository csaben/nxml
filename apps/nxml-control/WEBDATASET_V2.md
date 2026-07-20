# Immutable WebDataset v2

`nxml.hf-webdataset-snapshot.v2` is the training interchange contract for MIRA.
The cluster `nxml.segment-snapshot.v1` catalog remains authoritative. Hugging
Face stores durable, content-addressed publication batches and is never an edge
spool, receipt authority, or deletion signal.

Each segment is one immutable tar containing exactly MKV video, action Parquet,
and event Parquet members. The manifest records the outer and member sizes and
SHA-256 values, episode close identity, split, segment sequence, shared monotonic
clock, half-open timeline, global frame origin/count, controller/model lineage,
and artifact-probed codec lineage. A publication is rejected on a missing member,
checksum mismatch, unsafe path, schema mismatch, segment gap/overlap, split or
clock change within an episode, frame-ordinal gap, incompatible codec, or decoded
frame/action-row mismatch.

The compact compatibility profile is H.264 Main level 3.2 in Matroska, YUV420P,
1280x720, exact 60/1 nominal and average frame rates, GOP 60, and zero B-frames.
Time base and measured bitrate must be recorded from `ffprobe`; configured values
are not evidence. Image conversion is aspect-preserving padding only.

Distributed iteration assigns whole episodes by deterministic striding across
`rank * workers_per_rank + worker_id`. Segment boundaries therefore never split
an episode between consumers. MIRA windows use exact global frame ordinals and
may span adjacent content-addressed shards. Sticks are mean-pooled and buttons
OR-pooled over the declared source stride.

Hub consumers pin an immutable 40-character revision. Downloads go through a
bounded local cache: partial files are removed on restart, complete objects are
reused only after checksum verification, and least-recently-used unprotected
shards are evicted. One shard larger than the configured bound fails closed.
`HF_XET_CACHE` must point to measured local SSD storage.

Canaries are synthetic and `split=canary`; they cannot become gameplay training
input. Real cluster objects are published only from an immutable eligible
snapshot in bounded batches, and verified Hub visibility never authorizes cluster
deletion. `finetune_from` creates a new immutable lineage from a pretrained
checkpoint; `continue_from` resumes the exact same job state. They are not
interchangeable.
