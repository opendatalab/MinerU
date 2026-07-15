"""Centralized CLI output, error, and exit-code runtime."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

import httpx
import typer

from ..errors import MineruError, ServerNotRunningError, error_response
from . import output
from .contracts import (
    CliContext,
    CliGuidance,
    CliRenderer,
    CliResult,
    CliTaskResult,
    RenderableOutput,
    RichObject,
)

T = TypeVar("T")


def cli_ok(
    data: T | None = None,
    *,
    render: CliRenderer[T] | None = None,
    exit_code: int = 0,
    warnings: list[str] | None = None,
    notices: list[str] | None = None,
) -> CliResult[T]:
    return CliResult(
        data=data,
        render=render,
        exit_code=exit_code,
        warnings=warnings or [],
        notices=notices or [],
    )


def cli_task(
    data: T,
    *,
    status: str | None,
    render: CliRenderer[T] | None = None,
    fail_if_final_failed: bool,
    fail_statuses: frozenset[str] = frozenset({"failed"}),
    warnings: list[str] | None = None,
    notices: list[str] | None = None,
) -> CliTaskResult[T]:
    return CliTaskResult(
        data=data,
        render=render,
        warnings=warnings or [],
        notices=notices or [],
        status=status,
        fail_statuses=fail_statuses,
        fail_if_final_failed=fail_if_final_failed,
    )


def run_cli(
    ctx: CliContext,
    action: Callable[[], T | CliResult[T]],
    *,
    render: CliRenderer[T] | None = None,
    error_guidance: Callable[[MineruError], CliGuidance | None] | None = None,
    warnings: Callable[[T], list[str]] | None = None,
    notices: Callable[[T], list[str]] | None = None,
    exit_code: int | Callable[[T], int] = 0,
) -> None:
    try:
        value = action()
    except typer.Exit:
        raise
    except Exception as exc:
        guidance = _resolve_error_guidance(exc, error_guidance)
        emit_error(ctx, exc, guidance=guidance)
    result = _coerce_result(
        value,
        render=render,
        warnings=warnings,
        notices=notices,
        exit_code=exit_code,
    )
    emit_result(ctx, result)


def emit_result(ctx: CliContext, result: CliResult[T]) -> None:
    for notice in result.notices:
        output.print_notice(notice)
    for warning in result.warnings:
        output.print_notice(warning)

    if ctx.json_mode:
        if result.data is not None:
            output.print_json(result.data)

    elif result.data is not None:
        rendered_output: RenderableOutput | None
        if result.render is not None:
            rendered_output = result.render(result.data)
        else:
            rendered_output = str(result.data)
        if rendered_output is not None:
            _emit_rendered_output(rendered_output)

    exit_code = _result_exit_code(result)
    if exit_code:
        raise typer.Exit(exit_code)


def emit_error(ctx: CliContext, exc: Exception, *, guidance: CliGuidance | None = None) -> NoReturn:
    mineru_error = to_mineru_error(exc)
    if ctx.json_mode:
        payload = error_response(mineru_error)
        if guidance is not None:
            payload["guidance"] = guidance.data
        output.print_json(payload)
    else:
        output.print_error(mineru_error.message or str(exc))
        if guidance is not None:
            output.print_notice(guidance.text)
    raise typer.Exit(1) from None


def _emit_rendered_output(rendered_output: RenderableOutput) -> None:
    if isinstance(rendered_output, str):
        print(rendered_output)
        return
    if isinstance(rendered_output, RichObject):
        output.print_rich(rendered_output)
        return
    for item in rendered_output:
        if isinstance(item, str):
            print(item)
        elif isinstance(item, RichObject):
            output.print_rich(item)
        else:
            print(item)


def to_mineru_error(exc: Exception) -> MineruError:
    if isinstance(exc, MineruError):
        return exc
    if _is_connection_error(exc):
        return ServerNotRunningError()
    parsed = _parse_error_tuple(str(exc))
    if parsed is not None:
        code, message, param = parsed
        return MineruError(code, message, param)
    return MineruError("api_error", str(exc), None)


def _result_exit_code(result: CliResult[Any]) -> int:
    if (
        isinstance(result, CliTaskResult)
        and result.fail_if_final_failed
        and result.status is not None
        and result.status in result.fail_statuses
    ):
        return 1
    return result.exit_code


def _coerce_result(
    value: T | CliResult[T],
    *,
    render: CliRenderer[T] | None,
    warnings: Callable[[T], list[str]] | None,
    notices: Callable[[T], list[str]] | None,
    exit_code: int | Callable[[T], int],
) -> CliResult[T]:
    if isinstance(value, CliResult):
        return value
    resolved_exit_code = exit_code(value) if callable(exit_code) else exit_code
    return cli_ok(
        value,
        render=render,
        exit_code=resolved_exit_code,
        warnings=warnings(value) if warnings is not None else None,
        notices=notices(value) if notices is not None else None,
    )


def _resolve_error_guidance(
    exc: Exception,
    resolver: Callable[[MineruError], CliGuidance | None] | None,
) -> CliGuidance | None:
    if resolver is None:
        return None
    try:
        return resolver(to_mineru_error(exc))
    except Exception:
        return None


def _parse_error_tuple(raw: str) -> tuple[str, str, str | None] | None:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, tuple) or len(value) < 2 or not isinstance(value[0], str):
        return None
    param = value[2] if len(value) > 2 and isinstance(value[2], str) else None
    return value[0], str(value[1]), param


def _is_connection_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            ConnectionRefusedError,
            ConnectionResetError,
            ConnectionAbortedError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
        ),
    )
