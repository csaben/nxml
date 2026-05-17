"""Directory-of-JSON macro store.

One file per macro: ``<root>/<name>.json``. Names are sanitized to forbid
path separators, leading dots, and empty strings — anything else (including
unicode) is fine.
"""

from __future__ import annotations

import re
from pathlib import Path

from nx_macros.schema import Macro

_FORBIDDEN = re.compile(r"[/\\\x00]")


def sanitize_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("macro name must not be empty")
    if name.startswith("."):
        raise ValueError(f"macro name must not start with '.': {name!r}")
    if _FORBIDDEN.search(name):
        raise ValueError(f"macro name contains forbidden characters: {name!r}")
    return name


class MacroStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, name: str) -> Path:
        return self._root / f"{sanitize_name(name)}.json"

    def list(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.json") if p.is_file())

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def save(self, macro: Macro) -> Path:
        return macro.save(self._path(macro.name))

    def load(self, name: str) -> Macro:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"macro not found: {name!r} (looked at {path})")
        return Macro.load(path)

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
