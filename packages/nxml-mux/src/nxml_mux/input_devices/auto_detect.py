"""Auto-pick a mapper for a given evdev device.

Iterates the loaded registry and returns the first mapper whose
``device_match.name_regex`` matches the device's reported name. Caller is
expected to have populated the registry first (e.g. via
:func:`registry.load_bundled_mappers`).
"""

from __future__ import annotations

from nxml_mux.input_devices.mappers.base import Mapper
from nxml_mux.input_devices.registry import all_mappers, load_bundled_mappers


def detect_mapper_for_name(device_name: str, *, autoload_bundled: bool = True) -> Mapper | None:
    if autoload_bundled and not all_mappers():
        load_bundled_mappers()
    for mapper in all_mappers():
        if mapper.matches(device_name):
            return mapper
    return None


def detect_mapper_for_device(device_path: str, *, autoload_bundled: bool = True) -> Mapper | None:
    """Open the evdev device at ``device_path`` and dispatch to its mapper."""
    import evdev

    device = evdev.InputDevice(device_path)
    return detect_mapper_for_name(device.name, autoload_bundled=autoload_bundled)
