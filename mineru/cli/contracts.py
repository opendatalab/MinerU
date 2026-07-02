"""Shared CLI runtime contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from rich.console import ConsoleRenderable

T = TypeVar("T")
RichObject: TypeAlias = ConsoleRenderable
PlainObject: TypeAlias = str
RenderableObject: TypeAlias = RichObject | PlainObject
RenderableOutput: TypeAlias = RenderableObject | Iterable[RenderableObject]
CliRenderer = Callable[[T], RenderableOutput | None]


@dataclass(frozen=True)
class CliContext:
    json_mode: bool
    verbose: bool = False


@dataclass(frozen=True)
class CliResult(Generic[T]):
    data: T | None
    render: CliRenderer[T] | None = None
    exit_code: int = 0
    warnings: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CliTaskResult(CliResult[T]):
    status: str | None = None
    fail_statuses: frozenset[str] = frozenset({"failed"})
    fail_if_final_failed: bool = False
