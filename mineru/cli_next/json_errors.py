"""Helpers for consistent CLI JSON error output."""

from __future__ import annotations

import ast
from typing import NoReturn

import typer

from ..errors import MineruError, error_response
from .output import print_error, print_json


def to_mineru_error(exc: Exception, *, fallback_message: str | None = None) -> MineruError:
    if isinstance(exc, MineruError):
        return exc
    parsed = _parse_error_tuple(str(exc))
    if parsed is not None:
        code, message, param = parsed
        return MineruError(code, message, param)
    return MineruError("api_error", fallback_message or str(exc), None)


def exit_with_error(exc: Exception, *, json_mode: bool, fallback_message: str | None = None) -> NoReturn:
    mineru_error = to_mineru_error(exc, fallback_message=fallback_message)
    if json_mode:
        print_json(error_response(mineru_error))
    else:
        print_error(mineru_error.message or str(exc))
    raise typer.Exit(1) from None


def _parse_error_tuple(raw: str) -> tuple[str, str, str | None] | None:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, tuple) or len(value) < 2 or not isinstance(value[0], str):
        return None
    param = value[2] if len(value) > 2 and isinstance(value[2], str) else None
    return value[0], str(value[1]), param
