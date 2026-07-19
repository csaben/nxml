"""Deterministic training-row selection shared by stream and materialized readers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal

ControlSource = Literal["all", "human", "policy"]


def select_training_rows(
    rows: Iterable[Mapping[str, Any]], *, control_source: ControlSource = "all"
) -> Iterator[dict[str, Any]]:
    """Yield stable input order, filtering from v2 human_mask/ownership only.

    Human includes an explicit human input mask or any human-owned applied dimension.
    Policy includes any policy-owned applied dimension. Invalid rows never train.
    """
    for row in rows:
        if row.get("row_schema_id") == "nxml.dagger-actions.v2":
            from nxml_core.contracts import DaggerActionRecordV2, OwnershipCodeV2

            parsed = DaggerActionRecordV2.model_validate(row)
            if not parsed.bc_training_eligible:
                continue
            if control_source == "policy":
                continue
            if control_source == "human" and OwnershipCodeV2.HUMAN not in parsed.ownership:
                continue
            yield parsed.model_dump(mode="json")
            continue
        if not bool(row.get("valid", False)):
            continue
        mask = [bool(value) for value in row.get("human_action_mask", ())]
        ownership = [int(value) for value in row.get("ownership", ())]
        if control_source == "human" and not (any(mask) or 1 in ownership):
            continue
        if control_source == "policy" and 2 not in ownership:
            continue
        yield dict(row)


def iter_webdataset_rows(
    stream,
    *,
    control_source: ControlSource = "all",
    episode_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Read action Parquet members from a WebDataset tar in lexical member order."""
    import io
    import tarfile

    import pyarrow.parquet as pq

    with tarfile.open(fileobj=stream, mode="r:*") as archive:
        members = sorted(
            (
                item
                for item in archive.getmembers()
                if item.isfile()
                and item.name.endswith(".parquet")
                and not item.name.endswith(".events.parquet")
                and (
                    episode_ids is None
                    or item.name.removesuffix(".parquet").split("/")[-1] in episode_ids
                )
            ),
            key=lambda item: item.name,
        )
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read {member.name}")
            rows = pq.read_table(io.BytesIO(extracted.read())).to_pylist()
            yield from select_training_rows(rows, control_source=control_source)
