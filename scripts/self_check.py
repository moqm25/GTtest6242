"""Lightweight repository self-checks."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "paths.yml").exists(), "Missing configs/paths.yml"
    assert (root / "src" / "build_dataset.py").exists(), "Missing core build script"
    assert (root / "tests").is_dir(), "Missing tests directory"


if __name__ == "__main__":
    main()
