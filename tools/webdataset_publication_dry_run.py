#!/usr/bin/env python3
"""Fail-closed, offline publication eligibility check; performs no upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nxwm_mira.eval.domain_shift import publication_dry_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("inspections", type=Path)
    parser.add_argument("episode_ids", nargs="+")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    inspection_payload = json.loads(args.inspections.read_text())
    inspections = {item["segment_id"]: item for item in inspection_payload["segments"]}
    result = publication_dry_run(snapshot, set(args.episode_ids), inspections)
    print(
        json.dumps(
            {
                "schema_id": "nxml.webdataset-publication-dry-run.v1",
                "eligible": True,
                "episode_ids": result.episode_ids,
                "segment_ids": result.segment_ids,
                "mutation_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
