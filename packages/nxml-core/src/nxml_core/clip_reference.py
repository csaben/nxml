"""Canonical addressing for a temporal clip inside one immutable episode."""

from __future__ import annotations

from typing import Literal
from urllib.parse import parse_qs, quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClipReference(BaseModel):
    """Half-open episode clip on the episode's monotonic nanosecond clock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(min_length=1)
    start_timestamp_ns: int = Field(ge=0)
    end_timestamp_ns: int = Field(ge=0)
    clock: Literal["episode_monotonic_ns"] = "episode_monotonic_ns"

    @model_validator(mode="after")
    def valid_interval(self) -> ClipReference:
        if self.start_timestamp_ns >= self.end_timestamp_ns:
            raise ValueError("start_timestamp_ns must be less than end_timestamp_ns")
        return self

    def address(self) -> str:
        return (
            f"nxml-clip://{quote(self.episode_id, safe='')}"
            f"?start_timestamp_ns={self.start_timestamp_ns}"
            f"&end_timestamp_ns={self.end_timestamp_ns}"
            f"&clock={self.clock}"
        )

    @classmethod
    def parse_address(cls, address: str) -> ClipReference:
        parsed = urlparse(address)
        if parsed.scheme != "nxml-clip" or not parsed.netloc:
            raise ValueError("clip address must use nxml-clip://<episode_id>")
        query = parse_qs(parsed.query, strict_parsing=True)
        try:
            return cls(
                episode_id=unquote(parsed.netloc),
                start_timestamp_ns=int(query["start_timestamp_ns"][0]),
                end_timestamp_ns=int(query["end_timestamp_ns"][0]),
                clock=query["clock"][0],
            )
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"invalid clip address: {error}") from error
