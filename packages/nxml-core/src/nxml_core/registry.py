from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class _Entry:
    cls: type
    config_cls: type


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._entries: dict[str, _Entry] = {}

    def register(self, name: str, *, config_cls: type):
        def decorator(cls):
            if name in self._entries:
                raise ValueError(f"{self._name}: '{name}' already registered")
            self._entries[name] = _Entry(cls=cls, config_cls=config_cls)
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._entries:
            raise KeyError(f"{self._name}: '{name}' not found. Available: {sorted(self._entries)}")
        return self._entries[name].cls

    def get_config(self, name: str) -> type:
        return self._entries[name].config_cls

    def list_names(self) -> list[str]:
        return sorted(self._entries)
