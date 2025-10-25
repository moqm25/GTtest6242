"""Shared helpers for repository automation scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional


ANSI_CODES = {
    "reset": "\033[0m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def colorize(text: str, colour: str) -> str:
    """Return a string wrapped in ANSI colour codes (falls back to raw text)."""
    prefix = ANSI_CODES.get(colour, "")
    suffix = ANSI_CODES["reset"] if prefix else ""
    return f"{prefix}{text}{suffix}"


def run_command(
    command: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    env: Optional[dict[str, str]] = None,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command, returning the completed process."""
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=timeout,
        capture_output=capture_output,
        text=True,
        check=check,
    )
    return completed


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple Markdown table."""
    if not rows:
        return ""
    column_widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]
    header_line = " | ".join(f"{header:<{width}}" for header, width in zip(headers, column_widths))
    separator = " | ".join("-" * width for width in column_widths)
    data_lines = [
        " | ".join(f"{cell:<{width}}" for cell, width in zip(row, column_widths))
        for row in rows
    ]
    return "\n".join([header_line, separator, *data_lines])
