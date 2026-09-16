"""Error helpers for mineru-kit."""

from __future__ import annotations

from typing import NoReturn

import typer

from ..errors import MineruError
from .output import print_error


def exit_with_message(code: str, message: str, param: str | None = None) -> NoReturn:
    err = MineruError(code, message, param)
    print_error(err.message)
    raise typer.Exit(1) from None


__all__ = ["exit_with_message"]
