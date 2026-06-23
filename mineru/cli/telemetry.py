"""CLI telemetry context and first-run prompt."""

from __future__ import annotations

import sys

import typer

from ..doclib.client import DoclibClient
from ..doclib.telemetry import infer_default_client_context, reset_telemetry_context, set_telemetry_context

PROMPT_COMMANDS = {
    "parse",
    "read",
    "scan",
    "watch",
    "search",
    "find",
    "list",
    "show",
    "invalidate",
    "forget",
    "cleanup",
}


def prepare_cli_telemetry(ctx: typer.Context) -> None:
    token = set_telemetry_context(infer_default_client_context(source="cli"))
    ctx.call_on_close(lambda: reset_telemetry_context(token))
    command = _top_level_command(ctx)
    if command not in PROMPT_COMMANDS:
        return
    maybe_prompt_telemetry_consent()


def maybe_prompt_telemetry_consent() -> None:
    if not _is_interactive():
        return
    try:
        client = DoclibClient(timeout=5)
        status = client.get_telemetry_status()
    except Exception:
        return
    if status.state != "unset":
        return

    typer.echo("MinerU can send anonymous, aggregated telemetry to help improve doclib.")
    enabled = typer.confirm("Enable telemetry?", default=True)
    try:
        client.telemetry_action("enable" if enabled else "disable")
    except Exception:
        return


def _top_level_command(ctx: typer.Context) -> str | None:
    command = ctx.invoked_subcommand
    if command:
        return command
    args = list(getattr(ctx, "args", []) or [])
    return args[0] if args else None


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


__all__ = ["maybe_prompt_telemetry_consent", "prepare_cli_telemetry"]
