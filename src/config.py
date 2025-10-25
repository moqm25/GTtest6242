"""Helper utilities for loading project configuration files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Return the repository root."""
    return _ROOT


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=None)
def load_paths() -> Dict[str, str]:
    """Load canonical project paths from configs/paths.yml."""
    raw = _read_yaml(project_root() / "configs" / "paths.yml")
    resolved: Dict[str, str] = {}
    for key, value in raw.items():
        resolved[key] = str((project_root() / value).resolve())
    return resolved


def resolve_path(key: str) -> Path:
    """Resolve a configured path key to an absolute Path instance."""
    paths = load_paths()
    if key not in paths:
        raise KeyError(f"Unknown path key '{key}' in configs/paths.yml")
    return Path(paths[key])
