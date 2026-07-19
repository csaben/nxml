# Hugging Face storage architecture audit

## Recommendation

Keep `nxml-control` plus cluster object storage authoritative for ingest. A receipt is acknowledged only after the cluster object exists, its size and SHA-256 match, and SQLite registration commits. That receipt is the edge deletion gate: after receiving it, cradle-ns may delete both episode source files and its staged tar. Snapshot creation, materialization, quality dispositions, training, and deployment read cluster-owned objects and catalog state only; none may reach back to a cradle-ns path.

Use a private Hugging Face Storage Bucket as an asynchronous raw-shard replica, not the initial receipt gate. Buckets are S3-like Xet-backed object storage and avoid Git history for large changing collections. Keep a separate private Dataset repository for small immutable snapshot/catalog manifests and optional curated publishable WebDataset views. HF documents that buckets are mutable/non-versioned, whereas repositories retain version history; bucket-to-repository server-side copy is not currently available, so duplicating every raw shard into both would add avoidable transfer/storage cost. See the official [Storage Buckets comparison](https://huggingface.co/docs/hub/storage-buckets) and [bucket S3 API](https://huggingface.co/docs/hub/storage-buckets-s3).

The existing `apps/nxml-spool` directly uploads capture files to a Dataset repo and deletes after HF verification. Preserve it for compatibility, but do not use that path for the DAgger edge: it bypasses the cluster receipt/catalog/quality contract. `nxml-control` currently has a storage-neutral `ObjectStorage` protocol and an atomic local backend; it does not yet implement an HF bucket backend. `packages/nxml-core` only supports downloading individual `hf:` artifacts.

## Immutable mapping

Never overwrite a remote key, even though buckets permit it:

```text
HF bucket: <org>/nxml-raw
  datasets/<dataset_id>/shards/<shard_sha256>.tar
  receipts/<commit_id>.json

HF dataset repo: <org>/nxml-pokemon-za-v2-catalog
  snapshots/<snapshot_id_without_prefix>/manifest.json
  catalog/receipts/<commit_id>.json
  catalog/quality/<dataset_id>/<episode_id>/<disposition_id>.json
```

The cluster replication table should bind `commit_id`, `dataset_id`, `shard_id`, cluster `storage_key`, byte size, SHA-256, remote namespace/key, replication state, attempt count, error, and verified timestamp. A successful replica requires remote size/hash verification. Repository publication additionally records the exact immutable Hub commit SHA and path. Retry is idempotent by local commit ID plus destination. Replication failure never changes the committed receipt or training eligibility and never asks the edge to retain video.

Snapshots contain exact eligible episode IDs per shard plus recorded exclusions. A worker resolves those cluster object keys through the configured `ObjectStorage`, verifies each receipt digest, and reads only selected tar members. HF URIs are replication metadata, never cradle-ns source paths.

## Sharding and transfer defaults

Keep the existing target near 1 GiB per tar. Hugging Face describes large WebDatasets as many TAR shards, often around 1 GB, optimized for sequential streaming; see the official [WebDataset guide](https://huggingface.co/docs/hub/datasets-webdataset). Use deterministic content-addressed names, no compression around already-compressed video, and no more than two concurrent 1 GiB uploads initially. Increase to four only after measuring cluster disk I/O, uplink saturation, Xet cache pressure, and API errors. Configure a local-SSD `HF_XET_CACHE`; HF recommends this for cluster uploads and supports high-performance Xet mode, which intentionally consumes substantial CPU and bandwidth. See [upload behavior and Xet guidance](https://huggingface.co/docs/huggingface_hub/guides/upload).

For Dataset repositories, stay well below 100,000 files, 10,000 directory entries, 200 GB per file, and 100 files per commit; these are HF's current recommendations, not hard dataset size limits. Use nested hash-prefix directories if the catalog grows. See [repository storage limits](https://huggingface.co/docs/hub/storage-limits). HF Datasets supports iterative streaming without downloading an entire dataset and shard-aware worker distribution; see [dataset streaming](https://huggingface.co/docs/datasets/stream).

## Clark setup checklist

Do not create anything until Clark approves all of these:

- Owner namespace: personal account or organization; organization is recommended for durable service ownership.
- Private bucket name, recommended `<org>/nxml-raw`.
- Private Dataset repo name, recommended `<org>/nxml-pokemon-za-v2-catalog`.
- Confirm private/public policy. Recommended: private until gameplay rights, personal information, and redistribution are reviewed. Private repos are visible only to the owner/authorized organization members; see [repository visibility](https://huggingface.co/docs/hub/repositories-settings).
- Choose US or EU bucket/repository region where the plan supports it. Bucket server-side copies require the same region, and HF documents US/EU data residency; see [bucket security and regions](https://huggingface.co/docs/hub/storage-buckets-security).
- Confirm current plan/quota, expected monthly ingest, retention, deletion policy, and budget. Current official limits list 100 GB private storage for free accounts and larger paid allowances/overage; verify the live plan before upload at [HF storage limits](https://huggingface.co/docs/hub/storage-limits).
- Create a dedicated fine-grained token scoped only to write the approved bucket/repo. Organization policy may require fine-grained tokens; see [user access tokens](https://huggingface.co/docs/hub/security-tokens). Do not reuse a personal broad write token.
- Provision the token on cradle without chat/browser exposure:

  ```sh
  sudo install -d -o nxml-control -g nxml-control -m 0751 /etc/nxml-control
  sudo install -o nxml-control -g nxml-control -m 0640 /dev/null /etc/nxml-control/hf-token
  sudo setfacl -m u:arelius:r /etc/nxml-control/hf-token
  ```

  Enter the value interactively into that file, never as a command-line argument or committed environment value. Configuration may contain only `HF_TOKEN_FILE=/etc/nxml-control/hf-token`; the future replicator must read the file itself. Clark's operator-read preference is satisfied by the narrow ACL while the file remains non-world-readable.

HF supports creating private Dataset repos under a user or writable organization namespace, but creation is deliberately deferred; see the official [repository creation guide](https://huggingface.co/docs/huggingface_hub/guides/repository). No repo, bucket, credential, or upload was created during this audit.
