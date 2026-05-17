"""Mapper registry — track loaded mappers and look one up by id or device.

In-memory ``Mapper`` instances loaded explicitly via
:func:`load_bundled_mappers` (the YAMLs that ship with this package) or
:func:`register` for user-supplied ones.
"""

from __future__ import annotations

from pathlib import Path

from nxml_mux.input_devices.mappers.base import Mapper, load_mapper_dir

_BUNDLED_DIR = Path(__file__).parent / "mappers"
_REGISTRY: dict[str, Mapper] = {}


def register(mapper: Mapper) -> None:
    _REGISTRY[mapper.id] = mapper


def get(mapper_id: str) -> Mapper:
    if mapper_id not in _REGISTRY:
        raise KeyError(f"no mapper registered with id {mapper_id!r}")
    return _REGISTRY[mapper_id]


def all_mappers() -> list[Mapper]:
    return list(_REGISTRY.values())


def load_bundled_mappers() -> list[Mapper]:
    """Load every ``.yaml`` from this package's ``mappers/`` dir into the registry."""
    mappers = load_mapper_dir(_BUNDLED_DIR)
    for m in mappers:
        register(m)
    return mappers
