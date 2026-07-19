from __future__ import annotations

import pytest
from nxml_core import ClipReference
from pydantic import ValidationError


def test_clip_reference_has_canonical_round_trip_address() -> None:
    reference = ClipReference(
        episode_id="episode/with spaces",
        start_timestamp_ns=10,
        end_timestamp_ns=20,
    )
    assert reference.clock == "episode_monotonic_ns"
    assert ClipReference.parse_address(reference.address()) == reference


@pytest.mark.parametrize("start,end", [(10, 10), (11, 10)])
def test_clip_reference_rejects_empty_or_reverse_interval(start: int, end: int) -> None:
    with pytest.raises(ValidationError, match="less than"):
        ClipReference(
            episode_id="ep",
            start_timestamp_ns=start,
            end_timestamp_ns=end,
        )


def test_clip_reference_rejects_non_monotonic_clock() -> None:
    with pytest.raises(ValidationError, match="episode_monotonic_ns"):
        ClipReference(
            episode_id="ep",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
            clock="unix_seconds",  # type: ignore[arg-type]
        )
