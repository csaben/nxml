# Measured domain-shift gate

No codec or world-model finetune starts until a compact-codec gameplay snapshot
passes the immutable WebDataset validators and this audit. Static/neutral,
latency-contaminated, invalid, or explicitly vetoed episodes are infrastructure
evidence only and cannot satisfy this gate.

1. Freeze the published source snapshot ID, Hub revision, split, codec lineage,
   and exact checkpoint-99999 codec digest before evaluation.
2. Select matched, human-play validation windows from the legacy domain and the
   new compact domain. Stratify by scene motion, brightness, UI density, and
   codec/resolution; record counts and deterministic sample IDs.
3. Apply only aspect-preserving padding to the declared MIRA frame size. Never
   stretch. Use the manifest's exact frame ordinals and action pooling contract.
4. Measure frozen-checkpoint reconstruction L1, LPIPS, and DINO-feature distance
   on both domains, plus bitrate, decode failures, and temporal/frame alignment.
   Store per-stratum results and bootstrap confidence intervals in an immutable
   audit artifact linked to snapshot, Hub revision, git commit, and checkpoint.
5. Predeclare acceptance thresholds before examining the result. A material and
   statistically supported degradation on the compact domain authorizes a
   bounded 30--50k codec finetune using `finetune_from=checkpoint-99999`.
   Otherwise retain the frozen codec. `continue_from` is reserved for resuming
   the exact same job state and is never interchangeable with `finetune_from`.
6. Any world-model run must name the exact resulting codec digest. Promotion,
   deployment, AI arming, Pure AI, and multi-day training remain separate gates.

The next gate therefore requires a useful, non-neutral compact-codec gameplay
episode after the edge memory and input-isolation fixes, followed by strict
cluster reconstruction and an explicit training-eligible quality disposition.
