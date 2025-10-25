"""Lightweight console progress logger with elapsed time output."""

from __future__ import annotations

import sys
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Iterator, Optional, TypeVar

try:  # pragma: no cover - optional dependency
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:  # pragma: no cover - keep fallback lean
    Console = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = TextColumn = BarColumn = TimeElapsedColumn = TimeRemainingColumn = MofNCompleteColumn = None  # type: ignore[assignment]

F = TypeVar("F", bound=Callable[..., object])

@dataclass
class ProgressLogger:
    """Emit timestamped progress messages for long-running scripts."""

    label: str
    stream: object = sys.stdout

    def __post_init__(self) -> None:
        self._start = time.perf_counter()
        self._emit(f"{self.label} started")

    def _elapsed(self) -> float:
        return time.perf_counter() - self._start

    def _emit(self, message: str) -> None:
        timestamp = f"[{self._elapsed():6.1f}s]"
        print(f"{timestamp} {message}", file=self.stream, flush=True)

    def step(self, message: str, index: Optional[int] = None, total: Optional[int] = None) -> None:
        if index is not None and total is not None:
            message = f"[{index}/{total}] {message}"
        self._emit(message)

    def done(self, message: str = "Completed") -> None:
        self._emit(f"{message} in {self._elapsed():.1f}s")


def _use_rich() -> bool:
    if Console is None or Progress is None:
        return False
    force = os.environ.get("USE_RICH")
    if force == "1":
        return True
    if force == "0":
        return False
    return sys.stdout.isatty()


class _NoopAdvance:
    def advance(self, n: int = 1) -> None:  # noqa: D401 - simple noop helper
        """No-op advance method for fallbacks."""
        del n


@contextmanager
def timed_step(title: str, total: int | None = None):
    """
    Spinner if total is None; otherwise a progress bar with 'total'.
    Yields an object with .advance(n=1).
    """
    start = time.perf_counter()
    if _use_rich():
        console = Console()
        if total is None:
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn(f"[bold]{title}"),
                console=console,
                transient=True,
            ) as prog:
                task_id = prog.add_task(title, total=None)

                class _Adv:
                    def advance(self, n: int = 1) -> None:
                        del n  # spinner only

                try:
                    yield _Adv()
                finally:
                    prog.update(task_id, description=f"{title} • done")
        else:
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn(f"[bold]{title}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as prog:
                task_id = prog.add_task(title, total=total)

                class _Adv:
                    def advance(self, n: int = 1) -> None:
                        prog.update(task_id, advance=n)

                try:
                    yield _Adv()
                finally:
                    prog.update(task_id, completed=total)
        elapsed = time.perf_counter() - start
        console.print(f"[bold]{title}[/] done in {elapsed:.1f}s")
    else:
        pl = ProgressLogger(title)
        try:
            yield _NoopAdvance()
        finally:
            pl.done("done")


def progress_for_iterable(iterable: Iterable, label: str, length: int | None = None) -> Iterator:
    """
    Wrap an iterable with live progress updates. If length is unknown or Rich not available,
    gracefully yields without animation (or prints periodic ticks).
    """
    try:
        total = length if length is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
    except Exception:  # pragma: no cover
        total = None

    if _use_rich():
        console = Console()
        columns = [
            SpinnerColumn(spinner_name="dots"),
            TextColumn(f"[bold]{label}"),
        ]
        if total is not None:
            columns.extend(
                [
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ]
            )
        with Progress(*columns, console=console, transient=True) as prog:
            task_id = prog.add_task(label, total=total)
            for item in iterable:
                yield item
                if total is not None:
                    prog.update(task_id, advance=1)
            if total is None:
                prog.update(task_id, description=f"{label} • done")
        return

    pl = ProgressLogger(label)
    for i, item in enumerate(iterable, 1):
        yield item
        if i % 10000 == 0:
            pl.step(f"{i} items...")
    pl.done("done")


def long_task(label: str) -> Callable[[F], F]:
    def _wrap(fn: F) -> F:
        @wraps(fn)
        def _inner(*args, **kwargs):
            with timed_step(label):
                return fn(*args, **kwargs)

        return _inner  # type: ignore[return-value]

    return _wrap
